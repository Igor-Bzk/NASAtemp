from SREDT.utils import unpickle

features = ["koi_period", "koi_eccen", "koi_impact", "koi_duration", "koi_depth", "koi_ror", "koi_srho", "koi_prad", "koi_sma", "koi_incl", "koi_teq", "koi_insol", "koi_dor", "koi_model_snr", "koi_count", "koi_num_transits", "koi_bin_oedp_sig", "koi_steff", "koi_slogg", "koi_srad", "koi_smass", "koi_kepmag", "koi_fwm_stat_sig"]

labels = ["Conf", "FP"]

clf = unpickle("pkl/sredt_model_less_features_1.pkl")
clf.display("images/tree_less_features_1", features=features, labels=labels)