{
    name: "bert_qdmr_train_def_links",
    logdir: "logdir/%s" % self.name,
    model_config: "text2qdmr/configs/text2qdmr-base.jsonnet",
    model_config_args: {
        save_name: 'bert_qdmr_train_def_links',
        save_data_path: "text2qdmr/preproc_data/",
        spider_data_path: "data/spider/",
        break_data_path: "data/break/",
        grounding_path: "data/break/groundings/",
        grounding_mode: "first",
        train_max_values_from_database: 25,
        eval_max_values_from_database: 25,
        bs: 6,
        num_batch_accumulated: 4,
        pretrained_version: "bert",
        max_steps: 81000,
        num_layers: 8,
        lr: 7.44e-4,
        bert_lr: 3e-6,
        att: 1,
        end_lr: 0,
        update_name: 'relational_transformer',
        sc_link: true,
        merge_sc_link: true,
        cv_link: false,
        use_relations: true,
        use_graph_relations: false,
        use_type_relations: false,
        use_online_data_processing: true,
        num_dataloading_workers: 3, # half of the batch size
    },

    eval_name: "%s_beam_%d" % [self.name, self.eval_beam_size],
    eval_output: "__LOGDIR__/ie_dirs",
    eval_beam_size: 1,
    eval_steps: std.reverse([1000 * x for x in std.range(71, 81)]), 
    eval_section: ["val", "test"],
    vis_dir: "full_val",
    eval_tb_dir: "runs_viz/%s_%s" % [self.name, self.eval_section],
    eval_strict_decoding: true,
    # virtuoso_server: 'http://link_to_virtuoso_server/'
}