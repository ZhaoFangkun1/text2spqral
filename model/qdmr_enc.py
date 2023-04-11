import collections
import json
import os
import warnings
import copy

import attr
import numpy as np
from functools import lru_cache

import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig

from text2qdmr.model.modules import abstract_preproc
from text2qdmr.model.modules import encoder_modules
from text2qdmr.datasets.utils.spider_match_utils import (
    compute_schema_linking,
    compute_cell_value_linking
)
from text2qdmr.utils import corenlp1
from text2qdmr.utils import registry
from text2qdmr.utils import serialization
from text2qdmr.utils.serialization import ComplexEncoder, ComplexDecoder

from text2qdmr.datasets.qdmr import QDMRStepArg
from qdmr2sparql.structures import QdmrInstance, GroundingKey

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@attr.s
class EncoderState:
    state = attr.ib()
    memory = attr.ib()
    question_memory = attr.ib()
    schema_memory = attr.ib()
    words = attr.ib()

    pointer_memories = attr.ib()
    pointer_maps = attr.ib()

    m2c_align_mat = attr.ib()
    m2t_align_mat = attr.ib()

    value_memories = attr.ib()
    values = attr.ib()
    value_emb = attr.ib()

    grnd_idx = attr.ib()

    def find_word_occurrences(self, word):
        return [i for i, w in enumerate(self.words) if w == word]


@attr.s
class PreprocessedSchema:
    column_names = attr.ib(factory=list)
    table_names = attr.ib(factory=list)
    table_bounds = attr.ib(factory=list)
    column_to_table = attr.ib(factory=dict)
    table_to_columns = attr.ib(factory=dict)
    foreign_keys = attr.ib(factory=dict)
    foreign_keys_tables = attr.ib(factory=lambda: collections.defaultdict(set))
    primary_keys = attr.ib(factory=list)

    # only for bert version
    normalized_column_names = attr.ib(factory=list)
    normalized_table_names = attr.ib(factory=list)


def preprocess_schema_uncached(schema,
                               tokenize_func,
                               include_table_name_in_column,
                               fix_issue_16_primary_keys,
                               delete_first=False,
                               shift=False,
                               pretrained_type='bert'):
    """If it's bert, we also cache the normalized version of
    question/column/table for schema linking"""
    r = PreprocessedSchema()

    last_table_id = None

    start_idx = 1 if delete_first else 0  # 0
    col_shift = len(schema.tables) if shift else 0  # 3

    for i, column in enumerate(schema.columns[start_idx:]):
        if pretrained_type == 'bert':
            col_toks = tokenize_func(column.name, column.unsplit_name)
        elif pretrained_type == 'roberta' or pretrained_type == 'grappa':
            col_toks, col_idx_map = tokenize_func(column.name, column.unsplit_name)
        else:
            raise Exception(f'Unkonwn pretrained type {pretrained_type}')

        # assert column.type in ["text", "number", "time", "boolean", "others"]
        type_tok = f'<type: {column.type}>'

        # for bert, we take the representation of the first word
        column_name = col_toks
        if include_table_name_in_column:
            if column.table is None:
                table_name = ['<any-table>']
            else:
                table_name = tokenize_func(
                    column.table.name, column.table.unsplit_name)
            column_name += ['<table-sep>'] + table_name
        column_name += [type_tok]

        r.normalized_column_names.append(Bertokens(col_toks))


        r.column_names.append(column_name)

        table_id = None if column.table is None else column.table.id
        r.column_to_table[str(i + col_shift)] = table_id
        if table_id is not None:
            columns = r.table_to_columns.setdefault(str(table_id), [])
            columns.append(i + col_shift)
        if last_table_id != table_id:
            r.table_bounds.append(i + col_shift)
            last_table_id = table_id

        if column.foreign_key_for is not None:
            r.foreign_keys[str(column.id - start_idx + col_shift)] = column.foreign_key_for.id - start_idx + col_shift
            r.foreign_keys_tables[str(column.table.id)].add(column.foreign_key_for.table.id)

    r.table_bounds.append(len(schema.columns[start_idx:]))
    assert len(r.table_bounds) == len(schema.tables) + 1

    for i, table in enumerate(schema.tables):

        table_toks = tokenize_func(table.name, table.unsplit_name)
        r.normalized_table_names.append(Bertokens(table_toks))

        r.table_names.append(table_toks)
    last_table = schema.tables[-1]

    r.foreign_keys_tables = serialization.to_dict_with_sorted_values(r.foreign_keys_tables)
    r.primary_keys = [
        column.id - start_idx + col_shift
        for table in schema.tables
        for column in table.primary_keys
    ] if fix_issue_16_primary_keys else [
        column.id - start_idx + col_shift
        for column in last_table.primary_keys
        for table in schema.tables
    ]

    return r

def get_value_unit_dict(item, tokenizer, pretrained_type='bert', shuffle_values=False):
    assert len(item.values) == 1
    value_units = item.values[0]
    # group by tokens
    value_unit_by_toks = collections.defaultdict(list)
    for val_unit in value_units:

        # tokenize

        val_tokens = tokenizer(str(val_unit))
        val_tokens = str(val_unit).lower().split(' ')
        bert_tokens = Bertokens(val_tokens)

        value_unit_by_toks[repr(val_tokens)].append(
            attr.evolve(val_unit, tokenized_value=val_tokens, bert_tokens=bert_tokens))

    value_unit_by_toks_keys = list(value_unit_by_toks.keys())  # 将value_unit按照val分类

    if shuffle_values:
        num_items = len(value_unit_by_toks_keys)
        item_shuffle_order = torch.randperm(num_items)  # using random form torch to have worker seeds under control
        value_unit_by_toks_keys = [value_unit_by_toks_keys[i] for i in item_shuffle_order]
    value_unit_dict = {}
    for i, value_unit_key in enumerate(value_unit_by_toks_keys):

        value_units = value_unit_by_toks[value_unit_key]
        cur_idx = [0]
        cur_value_unit = attr.evolve(value_units[0], idx=i)
        str_value = str(cur_value_unit)
        assert str_value not in value_unit_dict, (value_unit_dict, cur_value_unit)
        values, str_values = [cur_value_unit], [str_value]
        idx_match = cur_value_unit.q_match['idx_question'] if cur_value_unit.q_match else None  # 对应问题的id
        type_match = cur_value_unit.q_match['match'] if cur_value_unit.q_match else None  # match-type

        for val_unit in value_units[1:]:
            assert cur_value_unit.tokenized_value == val_unit.tokenized_value, (val_unit, str_values, values)
            val_unit = attr.evolve(val_unit, idx=i)  # 改idx 可是改了之后id就一样了

            res = compare_value_units(cur_value_unit, val_unit)
            if res is None:
                # both are from schema or qdmr
                cur_idx.append(len(str_values))
                str_values.append(str(val_unit))
                values.append(val_unit)
            elif res is False:
                for idx in cur_idx[::-1]:
                    del str_values[idx]
                    del values[idx]
                cur_idx = [len(str_values)]
                cur_value_unit = val_unit
                str_values.append(str(cur_value_unit))
                values.append(cur_value_unit)

            if val_unit.q_match:
                if idx_match:  # 要最短
                    left = max(val_unit.q_match['idx_question'][0], idx_match[0])
                    right = min(val_unit.q_match['idx_question'][-1], idx_match[-1])
                    idx_match = tuple(range(left, right + 1))
                    type_match = 'VEM' if type_match == 'VEM' or val_unit.q_match['match'] == 'VEM' else 'VPM'
                else:
                    idx_match = val_unit.q_match['idx_question']
                    type_match = val_unit.q_match['match']

        if idx_match:
            for i, val_unit in enumerate(values):
                if val_unit.source == 'qdmr' and not val_unit.q_match:
                    values[i] = attr.evolve(val_unit, q_match={'idx_question': idx_match, 'match': type_match})
        value_unit_dict[tuple(str_values)] = values

    return value_unit_dict


def compare_value_units(value_unit1, value_unit2):
    # filter only units from text that are not numbers
    if value_unit1.source != 'text' and value_unit2.source != 'text':
        return
    elif value_unit1.source == 'text' and value_unit2.source != 'text':
        if value_unit1.value_type == 'number':
            return
        return False
    elif value_unit1.source != 'text' and value_unit2.source == 'text':
        if value_unit2.value_type == 'number':
            return
        return True
    else:
        assert value_unit1.q_match and value_unit2.q_match, (value_unit1, value_unit2)
        assert value_unit1.source == value_unit2.source == 'text', (value_unit1, value_unit2)
        if len(value_unit1.orig_value) <= len(value_unit2.orig_value):
            return True
        else:
            return False


def get_tokenized_values(value_unit_dict):
    tokenized_values = []

    for val_units in value_unit_dict.values():
        tokenized_values.append(val_units[0].tokenized_value)

    return [tokenized_values]


class BreakFullEncoderBertPreproc(abstract_preproc.AbstractPreproc):

    def __init__(
            self,
            save_path,
            db_path,
            fix_issue_16_primary_keys=False,
            include_table_name_in_column=False,
            pretrained_version="bert",
            compute_sc_link=True,
            compute_cv_link=False,
            use_bert_unlimited_length=False,
            use_column_type=False,
            use_general_grounding=True,
            use_graph_relations=False,
            use_type_relations=False,
            merge_sc_link=False,
            add_cellmatch=False,
            construct_general_grounding=False,
            use_bert_masks=False):

        self.data_dir = os.path.join(save_path, 'enc')
        self.db_path = db_path
        self.texts = collections.defaultdict(list)
        self.fix_issue_16_primary_keys = fix_issue_16_primary_keys
        self.include_table_name_in_column = include_table_name_in_column
        self.compute_sc_link = compute_sc_link
        self.compute_cv_link = compute_cv_link
        self.use_bert_unlimited_length = use_bert_unlimited_length
        self.use_column_type = use_column_type
        self.use_bert_masks = use_bert_masks
        self.pretrained_version = pretrained_version

        self.counted_db_ids = set()
        self.preprocessed_schemas = {}

        if self.pretrained_version == 'bert':
            self.pretrained_modelname = 'bert-base-uncased'
        elif self.pretrained_version == 'grappa':
            self.pretrained_modelname = 'Salesforce/grappa_large_jnt'

        elif self.pretrained_version == 'roberta':
            self.pretrained_modelname = 'roberta-large'

        self.config = AutoConfig.from_pretrained(self.pretrained_modelname)
        try:
            # add add_prefix_space=True to better deal with the Roberta tokenizers
            # otherwise there will be no special symbol chr(288) at the beginnings of words
            self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_modelname, add_prefix_space=True)
        except Exception as e:
            print("WARNING: could not run the tokenizer normally, seeing this error:", e)
            print("Trying to run with local_files_only=True")
            self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_modelname, local_files_only=True,
                                                           add_prefix_space=True)

        # TODO: should get types from the data
        column_types = ["text", "number", "time", "boolean", "others"]
        new_tokens = [f"<type: {t}>" for t in column_types]
        if include_table_name_in_column:
            new_tokens += ['<table-sep>', '<any-table>']
        self.tokenizer.add_tokens(new_tokens)

        self.use_general_grounding = use_general_grounding
        self.merge_sc_link = merge_sc_link
        self.add_cellmatch = add_cellmatch
        self.construct_general_grounding = construct_general_grounding
        self.use_graph_relations = use_graph_relations
        self.use_type_relations = use_type_relations

    def _tokenize(self, presplit, unsplit, pretokenized=None):
        if self.tokenizer:
            if pretokenized:
                all_toks = []
                idx_map = {}
                for i, token in enumerate(pretokenized):
                    toks = self.tokenizer.tokenize(token)
                    idx_map[i] = len(all_toks)
                    all_toks += toks
                idx_map[len(pretokenized)] = len(all_toks)
                return all_toks, idx_map
            elif presplit and self.pretrained_version != 'bert':
                all_toks = []
                idx_map = {}
                for i, token in enumerate(presplit):
                    toks = self.tokenizer.tokenize(token)
                    idx_map[i] = len(all_toks)
                    all_toks += toks
                idx_map[len(presplit)] = len(all_toks)
                return all_toks, idx_map
            toks = self.tokenizer.tokenize(unsplit)
            return toks
        return presplit

    def values_to_sc_link(self, sc_link, value_unit_dict, idx_map, schema):
        for val_units in value_unit_dict.values():
            for val_unit in val_units:
                if (not self.merge_sc_link or self.merge_sc_link and self.add_cellmatch) and val_unit.column:
                    matched = False
                    for column in schema.columns:
                        if column.orig_name == val_unit.column and column.table.orig_name == val_unit.table:
                            assert not matched
                            sc_link['col_val_match'][f"{column.id},{val_unit.idx}"] = 'CELLMATCH'
                            matched = True
                    assert matched, (val_unit, schema.columns)

                if val_unit.q_match:
                    for idx in val_unit.q_match['idx_question']:
                        for q_id in range(idx_map[idx], idx_map[idx + 1]):
                            link = sc_link['q_val_match'].get(f"{q_id},{val_unit.idx}")
                            if link == 'VEM' and val_unit.q_match['match'] == 'VPM':
                                continue
                            else:
                                sc_link['q_val_match'][f"{q_id},{val_unit.idx}"] = val_unit.q_match['match']

        return sc_link

    def recompute_sc_link(self, sc_link, m_type, shift_i=0, shift_j=0):
        _match = {}
        for ij_str in sc_link[m_type].keys():
            i_str, j_str = ij_str.split(",")
            i_str, j_str = int(i_str), int(j_str)
            _match[f"{i_str + shift_i},{j_str + shift_j}"] = sc_link[m_type][ij_str]
        sc_link[m_type] = _match
        return sc_link

    def add_item(self, item, section, idx_to_add, validation_info):
        preprocessed = self.preprocess_item(item, idx_to_add, validation_info)
        self.texts[section].append(preprocessed)

    def clear_items(self):
        self.texts = collections.defaultdict(list)

    def preprocess_item(self, item, idx_to_add, validation_info):
        value_unit_dict = validation_info

        if item.orig_spider_entry is not None:
            raw_question = item.orig_spider_entry['question']
        else:
            raw_question = item.text
        question, idx_map = self._tokenize(item.text_toks, raw_question, item.text_toks_for_val)


        all_tokenized_values = get_tokenized_values(value_unit_dict)

        has_schema = item.schema is not None
        if has_schema:
            preproc_schema = self._preprocess_schema(item.schema,
                                                     delete_first=self.construct_general_grounding or self.use_general_grounding,
                                                     shift=self.use_general_grounding)

            question_bert_tokens = Bertokens(question)


            sc_link = question_bert_tokens.bert_schema_linking(
                preproc_schema.normalized_column_names,
                preproc_schema.normalized_table_names,
                value_unit_dict
            )

            if value_unit_dict:
                sc_link = self.values_to_sc_link(sc_link, value_unit_dict, idx_map, item.schema)

        else:
            preproc_schema = None
            sc_link = {"q_col_match": {}, "q_tab_match": {}, "q_val_match": {}, "col_val_match": {}}
            if value_unit_dict:
                sc_link = self.values_to_sc_link(sc_link, value_unit_dict, idx_map, None)

        general_grounding = []
        general_grounding_types = {}
        if has_schema:
            general_grounding += preproc_schema.table_names  # 'table_names': ['department', 'head', 'management']
            tab_len = len(general_grounding)
            general_grounding_types['table'] = tab_len

            sc_link = self.recompute_sc_link(sc_link, 'q_col_match', shift_j=tab_len) # 将q_col_match中的列向右平移tbl_len个单位

            general_grounding += [c[:-1] for c in preproc_schema.column_names]
            general_grounding_types['column'] = len(preproc_schema.column_names)

            sc_link = self.recompute_sc_link(sc_link, 'col_val_match', shift_i=tab_len - 1,
                                                 shift_j=len(general_grounding))
            sc_link = self.recompute_sc_link(sc_link, 'q_val_match', shift_j=len(general_grounding))

        assert len(all_tokenized_values) == 1, len(all_tokenized_values)
        if all_tokenized_values[0]:
            general_grounding += all_tokenized_values[0]  # one grounding
            general_grounding_types['value'] = len(all_tokenized_values[0])

        assert len(general_grounding) == sum(list(general_grounding_types.values())), (
            general_grounding, general_grounding_types)


        cv_link = {"num_date_match": {}, "cell_match": {}}
        return {
            'raw_question': raw_question,
            'question': question,
            'db_id': item.schema.db_id if item.schema is not None else None,
            'sc_link': sc_link,
            'cv_link': cv_link,
            'columns': preproc_schema.column_names if has_schema and self.use_graph_relations else None,
            'tables': preproc_schema.table_names if has_schema and self.use_graph_relations else None,
            'tokenized_values': all_tokenized_values if not self.use_general_grounding else None,
            'values': None,
            'table_bounds': preproc_schema.table_bounds if has_schema and self.use_graph_relations else None,
            'column_to_table': preproc_schema.column_to_table if has_schema and self.use_graph_relations else None,
            'table_to_columns': preproc_schema.table_to_columns if has_schema and self.use_graph_relations else None,
            'foreign_keys': preproc_schema.foreign_keys if has_schema and self.use_graph_relations else None,
            'foreign_keys_tables': preproc_schema.foreign_keys_tables if has_schema and self.use_graph_relations else None,
            'primary_keys': preproc_schema.primary_keys if has_schema and self.use_graph_relations else None,
            'general_grounding': general_grounding,
            'general_grounding_types': general_grounding_types,
            'idx': item.subset_idx,
            'full_name': item.full_name,
            'subset_name': item.subset_name,
        }

    def validate_item(self, item, section,
                      shuffle_tables=False, shuffle_columns=False, shuffle_values=False,
                      shuffle_sort_dir=False,
                      shuffle_compsup_op=False,
                      shuffle_qdmr_ordering=False):

        has_schema = item.schema is not None

        if has_schema:
            if shuffle_tables or shuffle_columns:
                shuffled = self.shuffle_schema_inplace(item.schema, shuffle_tables, shuffle_columns)
                recompute_cache = shuffled
            else:
                recompute_cache = False

            preproc_schema = self._preprocess_schema(item.schema,
                                                     delete_first=self.construct_general_grounding or self.use_general_grounding,
                                                     shift=self.use_general_grounding,
                                                     recompute_cache=recompute_cache)  # preprocess schema

        if shuffle_qdmr_ordering:
            shuffled, item.qdmr_code, item.qdmr_args, item.qdmr_ops, item.grounding = \
                self.generate_random_topsort_qdmr(item.qdmr_code, item.qdmr_args, item.qdmr_ops, item.grounding)

        num_choices = len(item.values)
        assert num_choices == 1

        all_results = [True] * num_choices
        value_unit_dict = get_value_unit_dict(item, self.tokenizer.tokenize, pretrained_type=self.pretrained_version,
                                              shuffle_values=shuffle_values)

        if item.orig_spider_entry is not None:
            raw_question = item.orig_spider_entry['question']
        else:
            raw_question = item.text

        if shuffle_sort_dir:
            shuffled, item.qdmr_code, item.qdmr_args, item.qdmr_ops, item.grounding, raw_question, text_toks_for_val, value_unit_dict = \
                self.shuffle_qdmr_sort_dir(item.qdmr_code, item.qdmr_args, item.qdmr_ops, item.grounding, raw_question,
                                           item.text_toks_for_val, value_unit_dict)
            item.orig_spider_entry = None
            item.text = raw_question
            item.text_toks_for_val = text_toks_for_val

        if shuffle_compsup_op:
            shuffled, item.qdmr_code, item.qdmr_args, item.qdmr_ops, item.grounding, raw_question, text_toks_for_val, value_unit_dict = \
                self.shuffle_compsup_op(item.qdmr_code, item.qdmr_args, item.qdmr_ops, item.grounding, raw_question,
                                        item.text_toks_for_val, value_unit_dict)
            item.orig_spider_entry = None
            item.text = raw_question
            item.text_toks_for_val = text_toks_for_val

        if self.tokenizer and item.text_toks_for_val:
            question, idx_map = self._tokenize(item.text_toks, raw_question, item.text_toks_for_val)
        else:
            question = self._tokenize(item.text_toks, raw_question, item.text_toks_for_val)

        tokenized_values = get_tokenized_values(value_unit_dict)

        for i in range(num_choices):
            tokenized_values = tokenized_values[0]

            if has_schema:
                table_names = preproc_schema.table_names
                column_names = preproc_schema.column_names  # without '*'
                if not self.use_column_type:
                    column_names = [c[:-1] for c in column_names]
            else:
                column_names, table_names = [], []

            num_words = len(question) + 2 + \
                        sum(len(c) + 1 for c in column_names) + \
                        sum(len(t) + 1 for t in table_names) + \
                        sum(len(v) + 1 for v in tokenized_values)
            if not self.use_bert_unlimited_length and num_words > 512:
                print('Skipping {}: too long sequence {}'.format(item.full_name, num_words))
                all_results[i] = False  # remove long sequences

            return all_results, value_unit_dict

    def _preprocess_schema(self, schema, delete_first=False, shift=False, recompute_cache=False):
        if schema.db_id in self.preprocessed_schemas and not recompute_cache:
            return self.preprocessed_schemas[schema.db_id]
        result = preprocess_schema_uncached(schema, self._tokenize,
                                            self.include_table_name_in_column,
                                            self.fix_issue_16_primary_keys,
                                            pretrained_type=self.pretrained_version,
                                            delete_first=delete_first, shift=shift)
        self.preprocessed_schemas[schema.db_id] = result
        return result

    @classmethod
    def shuffle_qdmr_sort_dir(cls, qdmr_code, qdmr_args, qdmr_ops, grounding, raw_question, text_toks_for_val,
                              value_unit_dict):

        shuffled = False
        qdmr_code_cur = qdmr_code[0]
        qdmr_code_new = copy.deepcopy(qdmr_code_cur)

        for i_step, step in enumerate(qdmr_code_cur):
            op = step[0]
            args = step[1]
            if op.lower() == "sort" and len(args) == 3:
                sort_dir_arg = args[2]
                if not isinstance(sort_dir_arg, QDMRStepArg):
                    continue
                sort_dir_arg = sort_dir_arg.arg
                if not isinstance(sort_dir_arg, GroundingKey) or not sort_dir_arg.issortdir():
                    continue
                sort_dir_arg = sort_dir_arg.keys[0]

                # pattern "from ... to ... "
                def match_fromto_pattern(text_toks):
                    none_output = None, None, None

                    if "from" not in text_toks:
                        return none_output
                    index_from = text_toks.index("from")

                    if "to" not in text_toks[index_from:]:
                        return none_output
                    index_to = text_toks[index_from:].index("to") + index_from

                    if index_to == len(text_toks) - 1:
                        # 'to' is the last token - cannot do anything
                        return none_output

                    toks_replace_tgt = ["from"] + [text_toks[index_to + 1]] + ["to"] + text_toks[
                                                                                       index_from + 1: index_to]
                    tok_prefix_pos = index_from
                    tok_suffix_pos = index_to + 2
                    return tok_prefix_pos, tok_suffix_pos, toks_replace_tgt

                # direction keywords
                tok_replace_src = None
                toks_replace_tgt = None
                if sort_dir_arg == "ascending":
                    if "ascending" in text_toks_for_val:
                        tok_replace_src = "ascending"
                        toks_replace_tgt = ["descending"]
                    elif "increasing" in text_toks_for_val:
                        tok_replace_src = "increasing"
                        toks_replace_tgt = ["decreasing"]
                    elif "alphabetical" in text_toks_for_val:
                        tok_replace_src = "alphabetical"
                        toks_replace_tgt = ["descending", "alphabetical"]
                    elif "alphabetic" in text_toks_for_val:
                        tok_replace_src = "alphabetic"
                        toks_replace_tgt = ["descending", "alphabetic"]
                    elif "lexicographically" in text_toks_for_val:
                        tok_replace_src = "lexicographically"
                        toks_replace_tgt = ["reverse", "lexicographically"]
                elif sort_dir_arg == "descending":
                    if "descending" in text_toks_for_val:
                        tok_replace_src = "descending"
                        toks_replace_tgt = ["ascending"]
                    elif "decreasing" in text_toks_for_val:
                        tok_replace_src = "decreasing"
                        toks_replace_tgt = ["increasing"]
                else:
                    continue

                if tok_replace_src is not None:
                    # replace only one token
                    tok_prefix_pos = text_toks_for_val.index(tok_replace_src)
                    tok_suffix_pos = tok_prefix_pos + 1
                else:
                    tok_prefix_pos, tok_suffix_pos, toks_replace_tgt = match_fromto_pattern(text_toks_for_val)

                if toks_replace_tgt is not None and tok_prefix_pos is not None and tok_suffix_pos is not None:
                    # can replace augmentation
                    # get random number to decide
                    shuffle_this = torch.randint(2, (1,)).item()
                    if shuffle_this:
                        # fix value matching
                        for key, vals in value_unit_dict.items():
                            for val in vals:
                                if isinstance(val.q_match, dict) and "idx_question" in val.q_match:
                                    idxs = val.q_match["idx_question"]
                                    idxs_new = []
                                    for idx in idxs:
                                        if idx < tok_prefix_pos:
                                            idxs_new.append(idx)
                                        elif idx >= tok_prefix_pos and idx < tok_suffix_pos:
                                            # the value was matched right inside the tokens being changed - try to match to the same toekn in the new setup
                                            matched_token = text_toks_for_val[idx]
                                            if matched_token in toks_replace_tgt:
                                                idxs_new.append(tok_prefix_pos + toks_replace_tgt.index(matched_token))
                                        else:
                                            idxs_new.append(
                                                idx - (tok_suffix_pos - tok_prefix_pos) + len(toks_replace_tgt))
                                    val.q_match["idx_question"] = idxs_new

                        text_toks_for_val = text_toks_for_val[:tok_prefix_pos] + toks_replace_tgt + text_toks_for_val[
                                                                                                    tok_suffix_pos:]
                        sort_dir_arg = "descending" if sort_dir_arg == "ascending" else "ascending"
                        qdmr_code_new[i_step][1][2] = attr.evolve(qdmr_code_new[i_step][1][2],
                                                                  arg=GroundingKey.make_sortdir_grounding(
                                                                      ascending=(sort_dir_arg == "ascending")))
                        shuffled = True

        if shuffled:
            qdmr_args = None  # not updating grounding fow now (seems to not being used)
            qdmr_ops = None  # not updating grounding fow now (seems to not being used)
            grounding = None  # not updating grounding fow now (seems to not being used)
            raw_question = None  # not updating grounding fow now (seems to not being used)
            return True, [
                qdmr_code_new], qdmr_args, qdmr_ops, grounding, raw_question, text_toks_for_val, value_unit_dict
        else:
            return False, qdmr_code, qdmr_args, qdmr_ops, grounding, raw_question, text_toks_for_val, value_unit_dict

    @classmethod
    def shuffle_compsup_op(cls, qdmr_code, qdmr_args, qdmr_ops, grounding, raw_question, text_toks_for_val,
                           value_unit_dict):

        shuffled = False
        qdmr_code_cur = qdmr_code[0]
        qdmr_code_new = copy.deepcopy(qdmr_code_cur)

        op_kerwords_for_op = {}
        op_kerwords_for_op[">"] = ["larger", "bigger", "higher",
                                   "greater", "better",
                                   "more", "over", "after", "above"]
        op_kerwords_for_op["<"] = ["lower", "less", "smaller",
                                   "fewer", "worse", "below",
                                   "under", "before"]

        op_kerwords_for_op[">="] = ["larger than or equal".split(" "),
                                    "bigger than or equal".split(" "),
                                    "higher than or equal".split(" "),
                                    "more than or equal".split(" "),
                                    "at least".split(" ")]
        op_kerwords_for_op["<="] = ["smaller than or equal".split(" "),
                                    "less than or equal".split(" "),
                                    "fewer than or equal".split(" "),
                                    "lower than or equal".split(" "),
                                    "at most".split(" ")]

        op_kerwords_for_op["max"] = ["max", "maximum", "largest", "biggest", "highest", "most"]
        op_kerwords_for_op["min"] = ["min", "minimum", "smallest", "fewest", "lowest", "least"]

        op_substitute_for_op = {}
        op_substitute_for_op[">"] = [">", "<", ">=", "<="]
        op_substitute_for_op["<"] = [">", "<", ">=", "<="]
        op_substitute_for_op[">="] = [">", "<", ">=", "<="]
        op_substitute_for_op["<="] = [">", "<", ">=", "<="]
        op_substitute_for_op["min"] = ["min", "max"]
        op_substitute_for_op["max"] = ["min", "max"]

        for op, patterns in op_kerwords_for_op.items():
            for i_p, pattern in enumerate(patterns):
                if isinstance(pattern, str):
                    patterns[i_p] = (pattern,)
                elif isinstance(pattern, list):
                    patterns[i_p] = tuple(pattern)
                elif isinstance(pattern, tuple):
                    pass
                else:
                    raise RuntimeError(f"Unknown pattern {pattern} for op {op} of type {type(pattern)}")

        for i_step, step in enumerate(qdmr_code_cur):
            op = step[0]
            args = step[1]
            if op.lower() == "comparative" and len(args) == 3:
                comparative_arg = args[2]
                if not isinstance(comparative_arg, QDMRStepArg):
                    continue
                comparative_arg = comparative_arg.arg
                if not isinstance(comparative_arg, GroundingKey) or not comparative_arg.iscomp():
                    continue
                comparative_op = comparative_arg.keys[0]
            elif op.lower() == "superlative" and len(args) == 3:
                comparative_arg = args[0]
                if not isinstance(comparative_arg, QDMRStepArg):
                    continue
                comparative_arg = comparative_arg.arg
                if comparative_arg not in ["min", "max"]:
                    continue
                comparative_op = comparative_arg
            else:
                continue

            if comparative_op not in op_kerwords_for_op:
                continue

            # searching for the pattern in question tokens
            patterns_found = {}
            for i_pos in range(len(text_toks_for_val)):
                patterns_to_search = op_kerwords_for_op[comparative_op]
                for p in patterns_to_search:
                    if len(p) > len(text_toks_for_val) - i_pos:
                        # pattern won't fit at this position
                        continue

                    this_pattern_found = True
                    for i_tok, tok in enumerate(p):
                        if p[i_tok] != text_toks_for_val[i_pos + i_tok]:
                            this_pattern_found = False
                            break

                    if this_pattern_found:
                        patterns_found[(p, i_pos)] = True

            if len(patterns_found) != 1:
                continue

            (pattern, tok_prefix_pos), _ = list(patterns_found.items())[0]
            tok_suffix_pos = tok_prefix_pos + len(pattern)

            new_op_options = op_substitute_for_op[comparative_op]
            new_op = new_op_options[torch.randint(len(new_op_options), (1,)).item()]
            new_pattern_indx = torch.randint(len(op_kerwords_for_op[new_op]), (1,)).item()

            toks_replace_tgt = op_kerwords_for_op[new_op][new_pattern_indx]

            if toks_replace_tgt is not None and tok_prefix_pos is not None and tok_suffix_pos is not None:
                # can replace augmentation
                # fix value matching
                for key, vals in value_unit_dict.items():
                    for val in vals:
                        if isinstance(val.q_match, dict) and "idx_question" in val.q_match:
                            idxs = val.q_match["idx_question"]
                            idxs_new = []
                            for idx in idxs:
                                if idx < tok_prefix_pos:
                                    idxs_new.append(idx)
                                elif idx >= tok_prefix_pos and idx < tok_suffix_pos:
                                    # the value was matched right inside the tokens being changed - try to match to the same toekn in the new setup
                                    matched_token = text_toks_for_val[idx]
                                    if matched_token in toks_replace_tgt:
                                        idxs_new.append(tok_prefix_pos + toks_replace_tgt.index(matched_token))
                                else:
                                    idxs_new.append(idx - (tok_suffix_pos - tok_prefix_pos) + len(toks_replace_tgt))
                            val.q_match["idx_question"] = idxs_new

                text_toks_for_val = text_toks_for_val[:tok_prefix_pos] + list(toks_replace_tgt) + text_toks_for_val[
                                                                                                  tok_suffix_pos:]
                if isinstance(comparative_arg, GroundingKey):
                    if len(comparative_arg.keys) == 3:
                        new_grounding = GroundingKey.make_comparative_grounding(new_op, comparative_arg.keys[1],
                                                                                comparative_arg.keys[2])
                    else:
                        new_grounding = GroundingKey.make_comparative_grounding(new_op, comparative_arg.keys[1])
                else:
                    new_grounding = new_op

                if op == "comparative":
                    qdmr_code_new[i_step][1][2] = attr.evolve(qdmr_code_new[i_step][1][2], arg=new_grounding)
                elif op == "superlative":
                    qdmr_code_new[i_step][1][0] = attr.evolve(qdmr_code_new[i_step][1][0], arg=new_grounding)
                else:
                    raise RuntimeError(f"Do not know how to augment op {op}")
                shuffled = True

        if shuffled:
            qdmr_args = None  # not updating grounding fow now (seems to not being used)
            qdmr_ops = None  # not updating grounding fow now (seems to not being used)
            grounding = None  # not updating grounding fow now (seems to not being used)
            raw_question = None  # not updating grounding fow now (seems to not being used)
            return True, [
                qdmr_code_new], qdmr_args, qdmr_ops, grounding, raw_question, text_toks_for_val, value_unit_dict
        else:
            return False, qdmr_code, qdmr_args, qdmr_ops, grounding, raw_question, text_toks_for_val, value_unit_dict

    @classmethod
    def generate_random_topsort_qdmr(cls, qdmr_code, qdmr_args, qdmr_ops, grounding):

        qdmr_code_cur = qdmr_code[0]
        num_qdmr_steps = len(qdmr_code_cur)

        qdmr_adjacency_matrix = torch.zeros((num_qdmr_steps, num_qdmr_steps), dtype=torch.long)
        for i_step, step in enumerate(qdmr_code_cur):
            args = step[1]
            for arg in args:
                assert isinstance(arg,
                                  QDMRStepArg), f"Have arg {arg} at step {i_step} of parsed QDMR {qdmr_code_cur}, should have an instance of QDMRStepArg"
                if arg.arg_type == "ref":
                    assert len(arg.arg) == 1, f"Should have only one arg in QDMRStepArg of arg_type ref"
                    ref = arg.arg[0]
                    assert QdmrInstance.is_good_qdmr_ref(
                        ref), f"QDMR ref should be a str starting with # but have '{ref}' of type {type(ref)}"
                    i_ref = QdmrInstance.ref_to_index(ref)
                    qdmr_adjacency_matrix[i_ref, i_step] = 1
            # add special case for a ref in the arg of comparative
            if step[0] == "comparative" and len(step[1]) == 3 and \
                    step[1][2].arg is not None and step[1][2].arg.iscomp() and len(step[1][2].arg.keys) == 2 \
                    and QdmrInstance.is_good_qdmr_ref(step[1][2].arg.keys[1]):
                i_ref = QdmrInstance.ref_to_index(step[1][2].arg.keys[1])
                qdmr_adjacency_matrix[i_ref, i_step] = 1

        # select the order of elements
        new_node_id = {}
        for i_step in range(num_qdmr_steps):
            num_in_edges = qdmr_adjacency_matrix.sum(0)
            assert num_in_edges.min().item() == 0
            items_to_choose = (num_in_edges == 0).nonzero().view(-1)
            chosen_item = torch.randint(items_to_choose.numel(), (1,))
            chosen_item = items_to_choose[chosen_item]
            new_node_id[chosen_item.item()] = i_step
            # delete all edges from the chosen node
            qdmr_adjacency_matrix[chosen_item, :] = 0
            # block the chosen node form selection
            qdmr_adjacency_matrix[chosen_item, chosen_item] = num_qdmr_steps + 1

        shuffled = False
        for i_step in range(num_qdmr_steps):
            if new_node_id[i_step] != i_step:
                shuffled = True
                break

        if new_node_id[num_qdmr_steps - 1] != num_qdmr_steps - 1:
            warnings.warn(
                f"The last node should be last, otherwise smth is wrong, have permutation {new_node_id}, aborting this shuffle")
            shuffled = False

        if shuffled:
            # reorder steps
            qdmr_code_new = [None] * num_qdmr_steps
            for i_step in range(num_qdmr_steps):
                qdmr_code_new[new_node_id[i_step]] = copy.deepcopy(qdmr_code_cur[i_step])

            # fix qdmr refs
            for step in qdmr_code_new:
                args = step[1]
                for arg in args:
                    assert isinstance(arg,
                                      QDMRStepArg), f"Have arg {arg} at step {i_step} of parsed QDMR {qdmr_code_new}, should have an instance of QDMRStepArg"
                    if arg.arg_type == "ref":
                        assert len(arg.arg) == 1, f"Should have only one arg in QDMRStepArg of arg_type ref"
                        ref = arg.arg[0]
                        assert QdmrInstance.is_good_qdmr_ref(
                            ref), f"QDMR ref should be a str starting with # but have '{ref}' of type {type(ref)}"
                        i_ref = QdmrInstance.ref_to_index(ref)
                        arg.arg[0] = QdmrInstance.index_to_ref(new_node_id[i_ref])

                # add special case for a ref in the arg of comparative
                if step[0] == "comparative" and len(step[1]) == 3 and \
                        step[1][2].arg is not None and step[1][2].arg.iscomp() and len(step[1][2].arg.keys) == 2 \
                        and QdmrInstance.is_good_qdmr_ref(step[1][2].arg.keys[1]):
                    i_ref = QdmrInstance.ref_to_index(step[1][2].arg.keys[1])
                    key_list = list(step[1][2].arg.keys)
                    key_list[1] = QdmrInstance.index_to_ref(new_node_id[i_ref])
                    step[1][2].arg.keys = tuple(key_list)

            qdmr_args = None  # not updating grounding fow now (seems to not being used)
            qdmr_ops = None  # not updating grounding fow now (seems to not being used)
            grounding = None  # not updating grounding fow now (seems to not being used)
            return True, [qdmr_code_new], qdmr_args, qdmr_ops, grounding
        else:
            return False, qdmr_code, qdmr_args, qdmr_ops, grounding

    @classmethod
    def shuffle_schema_inplace(cls, schema, shuffle_tables, shuffle_columns):
        # this function shuffles the schema inplace:
        # the shuffled version will stay in memory for next batches

        # delete items which we are not maintaining
        schema.foreign_key_graph = None
        schema.orig = None

        num_tables = len(schema.tables)
        table_order = torch.randperm(num_tables) if shuffle_tables else torch.arange(num_tables)
        shuffled = (table_order != torch.arange(num_tables)).any().item()
        if shuffled:
            tables_new = []
            database_schema_table_names = []
            for i_table in range(num_tables):
                t = schema.tables[table_order[i_table]]
                t.id = i_table
                tables_new.append(t)
                database_schema_table_names.append(schema.database_schema.table_names[table_order[i_table]])
            schema.tables = tables_new
            schema.database_schema.table_names = database_schema_table_names

            # shuffle tables according to shuffled tables
            # first add columns with no tables
            columns_new = [col for col in schema.columns if col.table is None]
            for i_table in range(num_tables):
                for col in schema.columns:
                    if col.table is not None and col.table.id == i_table:
                        columns_new.append(col)
            # make ids of columns correspond to the order
            for i_col, col in enumerate(columns_new):
                col.id = i_col
            schema.columns = columns_new

        if shuffle_columns:
            # shuffling column inside each table
            for i_table in range(num_tables):
                num_cols = len(schema.tables[i_table].columns)
                min_id = min(col.id for col in schema.tables[i_table].columns)
                col_order = torch.randperm(num_cols)
                col_shuffled = (col_order != torch.arange(num_cols)).any().item()

                # sanity check for column ids
                col_ids = set(col.id for col in schema.tables[i_table].columns)
                assert col_ids == set(i_ + min_id for i_ in range(
                    num_cols)), f"Columns of table {schema.tables[i_table].orig_name} have wrong ids: {col_ids}"

                if col_shuffled:
                    shuffled = True
                    for i_col, col in enumerate(schema.tables[i_table].columns):
                        col.id = min_id + col_order[i_col].item()
                    schema.tables[i_table].columns = sorted(schema.tables[i_table].columns, key=lambda col: col.id)

                    # sort all the columns by ids
            schema.columns = sorted(schema.columns, key=lambda col: col.id)

        return shuffled

    def save(self, partition=None):
        os.makedirs(self.data_dir, exist_ok=True)
        if partition is None:
            self.tokenizer.save_pretrained(self.data_dir)

        self.config.save_pretrained(self.data_dir)

        for section, texts in self.texts.items():
            with open(os.path.join(self.data_dir, section + '.jsonl'), 'w') as f:
                for text in texts:
                    f.write(json.dumps(text, cls=ComplexEncoder) + '\n')

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.data_dir)

    def dataset(self, section):
        return [
            json.loads(line, cls=ComplexDecoder)
            for line in open(os.path.join(self.data_dir, section + '.jsonl'))]


@registry.register('encoder', 'text2qdmr')
class BreakFullEncoderBert(torch.nn.Module):
    Preproc = BreakFullEncoderBertPreproc
    batched = True

    def __init__(
            self,
            device,
            preproc,
            update_config={},
            bert_token_type=False,
            summarize_header="first",
            use_column_type=True,
            include_in_memory=('question', 'column', 'table'),
            use_relations=False):
        super().__init__()
        self._device = device
        self.preproc = preproc
        self.bert_token_type = bert_token_type
        self.base_enc_hidden_size = 768
        assert summarize_header in ["first", "avg"]
        self.summarize_header = summarize_header
        self.enc_hidden_size = self.base_enc_hidden_size
        self.use_column_type = self.preproc.use_column_type
        self.use_relations = use_relations

        self.include_in_memory = set(include_in_memory)
        update_modules = {
            'relational_transformer':
                encoder_modules.RelationalTransformerUpdate,
            'none':
                encoder_modules.NoOpUpdate,
        }

        self.encs_update = registry.instantiate(
            update_modules[update_config['name']],
            update_config,
            unused_keys={"name"},
            device=self._device,
            hidden_size=self.enc_hidden_size,
            sc_link=True,
        )

        self.bert_model = AutoModel.from_pretrained(self.preproc.pretrained_modelname)
        self.tokenizer = self.preproc.tokenizer
        self.bert_model.resize_token_embeddings(len(self.tokenizer))  # several tokens added
        self.use_bert_masks = self.preproc.use_bert_masks

    def forward(self, descs):
        batch_token_lists = []
        batch_id_to_retrieve_question = []
        batch_id_to_retrieve_grnd = []
        if self.summarize_header == "avg":
            batch_id_to_retrieve_grnd_2 = []
        long_seq_set = set()
        batch_id_map = {}  # some long examples are not included
        if not self.preproc.use_general_grounding:
            # sample one grounding
            all_grnd_idx = np.zeros(len(descs), dtype=np.int).tolist()
        for batch_idx, desc in enumerate(descs):
            qs = self.pad_single_sentence_for_bert(desc['question'], cls=True)  # cls + question + sep


            grnds = [self.pad_single_sentence_for_bert(g, cls=False) for g in desc['general_grounding']]
            token_list = qs + [g for grnd in grnds for g in grnd]

            # ['[CLS]', 'how', 'many', 'heads', 'of', 'the', 'departments', 'are', 'older', 'than', '56', '?',
            # '[SEP]', 'department', '[SEP]', 'head', '[SEP]', 'management', '[SEP]', 'department', 'id', '[SEP]',
            # 'name', '[SEP]', 'creation', '[SEP]', 'ranking', '[SEP]', 'budget', 'in', 'billions', '[SEP]', 'nu',
            # '##m', 'employees', '[SEP]', 'head', 'id       ', '[SEP]', 'name', '[SEP]', 'born', 'state', '[SEP]',
            # 'age', '[SEP]', 'department', 'id', '[SEP]', 'head', 'id', '[SEP]', 'temporary', 'acting', '[SEP]',
            # '1', 'january', '205', '##6', ',', '00', ':', '00', ':', '00', '[SEP]', '56', '[SEP]']
            assert self.check_bert_seq(token_list)
            if not self.preproc.use_bert_unlimited_length and len(token_list) > 512:
                print(f"{descs[batch_idx]['full_name']} is too long ({len(token_list)}) - skipping it")
                long_seq_set.add(batch_idx)
                continue

            q_b = len(qs)

            grnd_b = q_b + sum(len(g) for g in grnds)  # question_len + tbl_len+col_len
            # leave out [CLS] and [SEP]
            question_indexes = list(range(q_b))[1:-1]  # question在token list的下标

            grnd_indexes = np.cumsum([q_b] + [len(token_list) for token_list in grnds[:-1]]).tolist()  # 多单词以第一个单词为准

            if self.summarize_header == "avg":
                grnd_indexes_2 = np.cumsum([q_b - 2] + [len(token_list) for token_list in grnds]).tolist()[1:]

            # grnd_indexes_2=[13, 15, 17, 20, 22, 24, 26, 30, 34, 37, 39, 42, 44, 47, 50, 53, 64, 66]
            # table、column、value在token_list中的下标 多单词以最后一个单词为准

            indexed_token_list = self.tokenizer.convert_tokens_to_ids(token_list)
            # indexed_token_list=[101, 2129, 2116, 4641, 1997, 1996, 7640, 2024,
            # 3080, 2084, 5179, 1029, 102, 2533, 102, 2132, 102, 2968, 102, 2533, 8909, 102, 2171, 102, 4325, 102,
            # 5464, 102, 5166, 1999, 25501, 102, 16371, 2213, 5126, 102, 2132, 8909, 102, 2171, 102, 2141, 2110, 102,
            # 2287, 102, 2533, 8909, 102, 2132, 8909, 102, 5741, 3772, 102, 1015, 2254, 16327, 2575, 1010, 4002,
            # 1024, 4002, 1024, 4002, 102, 5179, 102]
            batch_token_lists.append(indexed_token_list)

            question_rep_ids = torch.LongTensor(question_indexes).to(self._device)  # (1,,,,,11)
            batch_id_to_retrieve_question.append(question_rep_ids)

            grnd_rep_ids = torch.LongTensor(grnd_indexes).to(self._device)  # 各table、column、value在token list中的下标
            batch_id_to_retrieve_grnd.append(grnd_rep_ids)
            if self.summarize_header == "avg":
                assert (all(i2 >= i1 for i1, i2 in zip(grnd_indexes, grnd_indexes_2))), descs[batch_idx][
                    'full_name']
                grnd_rep_ids_2 = torch.LongTensor(grnd_indexes_2).to(self._device)
                batch_id_to_retrieve_grnd_2.append(grnd_rep_ids_2)

            batch_id_map[batch_idx] = len(batch_id_map)

        if len(long_seq_set) == len(descs):  # 都太长
            # all batch elements al too long
            return [None] * len(descs)
        padded_token_lists, att_mask_lists, tok_type_lists, position_ids_lists = self.pad_sequence_for_bert_batch(
            batch_token_lists,
            use_bert_unlimited_length=self.preproc.use_bert_unlimited_length,
            use_bert_masks=self.use_bert_masks,
            indexes=batch_id_to_retrieve_grnd, descs=descs)
        tokens_tensor = torch.LongTensor(padded_token_lists).to(self._device)  # padding之后的token_lists
        att_masks_tensor = torch.LongTensor(att_mask_lists).to(self._device)  # mask 表示哪些是padding
        position_ids_tensor = torch.LongTensor(position_ids_lists).to(self._device)  # 位置信息

        if self.bert_token_type:
            tok_type_tensor = torch.LongTensor(tok_type_lists).to(self._device)
            bert_output = self.bert_model(tokens_tensor,
                                          attention_mask=att_masks_tensor, token_type_ids=tok_type_tensor,
                                          position_ids=position_ids_tensor)[0]
        else:

            bert_output = self.bert_model(tokens_tensor,
                                          attention_mask=att_masks_tensor, position_ids=position_ids_tensor)[0]

        enc_output = bert_output
        if not self.preproc.use_general_grounding:
            has_vals = [len(desc['tokenized_values'][grnd_idx]) for desc, grnd_idx in zip(descs, all_grnd_idx)]

        grnd_pointer_maps = [
            {
                i: [i]
                for i in range(len(desc['general_grounding']))
            }
            for desc in descs
        ]  # 是个列表 grnd_pointer_maps[i] 存储了第i个desc的general_grounding映射


        # the batched version of rat transformer
        q_enc_batch = []
        grnd_enc_batch = []
        relations_batch = []
        for batch_idx, desc in enumerate(descs):
            if batch_idx in long_seq_set:
                continue

            bert_batch_idx = batch_id_map[batch_idx]
            q_enc = enc_output[bert_batch_idx][batch_id_to_retrieve_question[bert_batch_idx]]  # question部分的enc

            grnd_enc = enc_output[bert_batch_idx][batch_id_to_retrieve_grnd[bert_batch_idx]]  # grnd部分的enc

            if self.summarize_header == "avg":
                grnd_enc_2 = enc_output[bert_batch_idx][batch_id_to_retrieve_grnd_2[bert_batch_idx]]

                grnd_enc = (grnd_enc + grnd_enc_2) / 2.0  # avg of first and last token

            grnd_boundaries = list(range(len(desc["general_grounding"]) + 1))
            assert q_enc.shape[0] == len(desc["question"])
            assert grnd_enc.shape[0] == grnd_boundaries[-1]

            total_enc_len = q_enc.shape[0] + grnd_enc.shape[0]
            if self.use_relations:
                general_grounding_types = desc.get('general_grounding_types')
                # "general_grounding_types": {"table": 3, "column": 13, "value": 2},
                input_types = general_grounding_types if not self.preproc.merge_sc_link else None
                relation_map = self.encs_update.relation_map

                # Catalogue which things are where
                relations = self.encs_update.compute_relations(
                    desc,
                    enc_length=total_enc_len,  # question + grnd
                    q_enc_length=q_enc.shape[0],
                    c_enc_length=grnd_enc.shape[0],
                    t_enc_length=None,
                    c_boundaries=grnd_boundaries,
                    t_boundaries=None,
                    input_types=input_types,
                    v_boundaries=None,
                    relation_map=relation_map)
            else:
                relations = np.zeros((total_enc_len, total_enc_len), dtype=np.int64)

            relations = torch.as_tensor(relations)
            q_enc_batch.append(q_enc)
            grnd_enc_batch.append(grnd_enc)
            relations_batch.append(relations)

        q_enc_new_item_batch, grnd_enc_new_item_batch, _, align_mat_item_batch = \
            self.encs_update.forward_batched(relations_batch, q_enc_batch, grnd_enc_batch)

        result = []
        num_processed_items = 0
        for batch_idx, desc in enumerate(descs):
            if batch_idx in long_seq_set:
                result.append(None)
                continue
            else:
                batch_idx = num_processed_items
                num_processed_items = num_processed_items + 1
            memory = []

            q_enc_new_item = q_enc_new_item_batch[batch_idx].unsqueeze(0)
            grnd_enc_new_item = grnd_enc_new_item_batch[batch_idx].unsqueeze(0)
            if align_mat_item_batch:
                align_mat_item = [align_mat_item_batch[0][batch_idx], align_mat_item_batch[1][batch_idx]]
            else:
                align_mat_item = (None, None)

            if 'question' in self.include_in_memory:
                memory.append(q_enc_new_item)
            if 'grounding' in self.include_in_memory:
                memory.append(grnd_enc_new_item)
            memory = torch.cat(memory, dim=1)

            result.append(EncoderState(
                state=None,
                memory=memory,
                question_memory=None,
                schema_memory=None,
                # TODO: words should match memory
                words=None,
                pointer_memories={'grounding': grnd_enc_new_item,},
                pointer_maps={'grounding': grnd_pointer_maps[batch_idx]},
                m2c_align_mat=align_mat_item[0],
                m2t_align_mat=align_mat_item[1],
                value_memories=None,
                values=None,
                value_emb=None,
                grnd_idx=0,
            ))

        return result

    def check_bert_seq(self, toks):
        if toks[0] == self.tokenizer.cls_token and toks[-1] == self.tokenizer.sep_token:
            return True
        else:
            return False

    def pad_single_sentence_for_bert(self, toks, cls=True):
        if cls:
            return [self.tokenizer.cls_token] + toks + [self.tokenizer.sep_token]
        else:
            return toks + [self.tokenizer.sep_token]

    def pad_sequence_for_bert_batch(self, tokens_lists, use_bert_unlimited_length=False, use_bert_masks=False,
                                    indexes=None, descs=None):
        pad_id = self.tokenizer.pad_token_id
        max_len = max([len(it) for it in tokens_lists])
        if not (use_bert_unlimited_length or use_bert_masks):
            assert max_len <= 512
        toks_ids = []
        att_masks = []
        tok_type_lists = []
        position_ids_lists = []
        for item_toks, desc, idx in zip(tokens_lists, descs, indexes):
            padded_item_toks = item_toks + [pad_id] * (max_len - len(item_toks))
            toks_ids.append(padded_item_toks)

            if use_bert_masks:
                _att_mask = torch.zeros((1, max_len, max_len), dtype=torch.int64)
                _position_ids_list = list(range(0, idx[0]))

                sep_id = list(idx) + [len(item_toks)]

                # all depends on question
                _att_mask[:, :, :sep_id[0]] += 1
                for start_block, end_block in zip(sep_id, sep_id[1:]):
                    _att_mask[:, start_block:end_block, start_block:end_block] += 1
                    _position_ids_list += list(range(idx[0], idx[0] + end_block - start_block))
                _position_ids_list += [511] * (max_len - len(item_toks))

            else:
                _att_mask = [1] * len(item_toks) + [0] * (max_len - len(item_toks))

            att_masks.append(_att_mask)

            first_sep_id = padded_item_toks.index(self.tokenizer.sep_token_id)  # 第一个sep
            assert first_sep_id > 0
            assert not use_bert_masks or first_sep_id == sep_id[0] - 1, (first_sep_id, sep_id[0])
            _tok_type_list = [0] * (first_sep_id + 1) + [1] * (max_len - first_sep_id - 1)  # 前半部分问题 后半部分grnd
            if use_bert_unlimited_length:
                _position_ids_list = list(range(0, first_sep_id + 1)) + [511] * (
                        max_len - first_sep_id - 1)  # 511 - maximum position Id for BERT
            elif use_bert_masks:
                assert len(_position_ids_list) == len(_tok_type_list), (len(_position_ids_list), len(_tok_type_list))
            else:
                _position_ids_list = list(range(0, len(_tok_type_list)))
            tok_type_lists.append(_tok_type_list)
            position_ids_lists.append(_position_ids_list)
        if use_bert_masks:
            att_masks = torch.cat(att_masks)
        return toks_ids, att_masks, tok_type_lists, position_ids_lists


@lru_cache(maxsize=100000)
def annotate_bertokens_with_corenlp(tok):
    ann = corenlp1.annotate(tok, annotators=['tokenize', 'ssplit', 'lemma'])
    lemmas = [tok.lemma.lower() for sent in ann.sentence for tok in sent.token]
    lemma_word = " ".join(lemmas)
    return lemma_word


class Bertokens:
    def __init__(self, pieces):
        self.pieces = pieces

        self.normalized_pieces = None
        self.recovered_pieces = None
        self.idx_map = None

        self.normalize_toks()

    def normalize_toks(self):
        """
        If the token is not a word piece, then find its lemma
        If it is, combine pieces into a word, and then find its lemma
        E.g., a ##b ##c will be normalized as "abc", "", ""
        NOTE: this is only used for schema linking
        """
        self.startidx2pieces = dict()
        self.pieces2startidx = dict()
        cache_start = None
        for i, piece in enumerate(self.pieces + [""]):
            if piece.startswith("##"):
                if cache_start is None:
                    cache_start = i - 1

                self.pieces2startidx[i] = cache_start
                self.pieces2startidx[i - 1] = cache_start
            else:
                if cache_start is not None:
                    self.startidx2pieces[cache_start] = i
                cache_start = None
        assert cache_start is None

        # combine pieces, "abc", "", ""
        combined_word = {}
        for start, end in self.startidx2pieces.items():
            assert end - start + 1 < 10
            pieces = [self.pieces[start]] + [self.pieces[_id].strip("##") for _id in range(start + 1, end)]
            word = "".join(pieces)
            combined_word[start] = word

        # remove "", only keep "abc"
        idx_map = {}
        new_toks = []
        for i, piece in enumerate(self.pieces):
            if i in combined_word:
                idx_map[len(new_toks)] = i
                new_toks.append(combined_word[i])
            elif i in self.pieces2startidx:
                # remove it
                pass
            else:
                idx_map[len(new_toks)] = i
                new_toks.append(piece)
        self.idx_map = idx_map

        # lemmatize "abc"
        normalized_toks = []
        for i, tok in enumerate(new_toks):

            lemma_word = annotate_bertokens_with_corenlp(tok)

            normalized_toks.append(lemma_word)

        self.normalized_pieces = normalized_toks
        self.recovered_pieces = new_toks

    def bert_schema_linking(self, columns, tables, value_unit_dict=None):
        question_tokens = self.normalized_pieces

        column_tokens = [c.normalized_pieces for c in columns]
        table_tokens = [t.normalized_pieces for t in tables]

        if value_unit_dict and not (len(value_unit_dict) == 1 and value_unit_dict.get(('<UNK>',))):  # 有 有效的 value_unit
            value_tokens = [v[0].bert_tokens.normalized_pieces for v in value_unit_dict.values()]

        else:
            value_tokens = None
        sc_link = compute_schema_linking(question_tokens, column_tokens, table_tokens, value_tokens)

        new_sc_link = {}  # 一个单词出现多次 按第一次出现的下标
        for m_type in sc_link:
            _match = {}
            for ij_str in sc_link[m_type]:
                q_id_str, col_tab_id_str = ij_str.split(",")
                q_id, col_tab_id = int(q_id_str), int(col_tab_id_str)
                real_q_id = self.idx_map[q_id]
                _match[f"{real_q_id},{col_tab_id}"] = sc_link[m_type][ij_str]

            new_sc_link[m_type] = _match
        return new_sc_link



