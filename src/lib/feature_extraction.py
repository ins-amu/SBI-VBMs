import numpy as np
# from numba import jit
from scipy import signal
from scipy.stats import iqr
import scipy.stats as stats
from numpy import linalg as LA
from copy import copy, deepcopy
from src.utility import flatten
from scipy.signal import hilbert
from src.lib.statistics import fcd_filt as _fcd_filt
from sklearn.decomposition import PCA
from src.lib.utility import build_matrix
from src.utility import preprocessing_signal
from scipy.stats import moment, skew, kurtosis
from src.lib.utility import filter_butter_bandpass
from src.lib.utility import compute_plv, compute_pli
from src.lib.phFCD import calc_FCD as calc_FCD_ph
from src.tools import set_k_diogonal, select_upper_triangular
from src.lib.edgeFCD import local_to_global_coherence, go_edge
from src.lib.statistics import extract_fcd_filt


class Features:
    '''
    Feature calculation functions
    '''

    available_features = [
        'moments',
        'coactivation',
        'coactivation_ph',
        'higher_moments',
        'spectral_power',
        'envelope',
        'fcd_corr',
        'phFCD',
        'fcd_elements',
        'fcd_corr_regions',
        'fcd_regions',
        'fcd_regions_filt',
        'fcd_filt',
        'fc_corr_regions',
        'fc_corr',
        'fc_elements',
        'fc_sum',
        'fc_homotopic',
        'fcd_sum',
        'fcd_edge',
        "fcd_edge_var",
        'raw_ts',
        'PSD_under_area',
        'PSD_under_power',
        'max_PSD_freq',
        'abs_moments',
        'high_low_mu',
        'burstiness',
        'CustomSkewness',
        'IQR',
        'entropy',
        'fluidity',
        "spectral_connectivity",
        "KOP",
        "local_to_global_coherence",
        "preprocessing",
        # "fcd_mask",
        "fcd_filt_limbic",
        "fcd_filt_frontal",
        "fcd_filt_temporal",
        "fcd_filt_occipital",
        "fcd_filt_parietal",
        "fcd_filt_centralstructures",
        "fcd_filt_0",
        "fcd_filt_1",
        "fcd_filt_2",
        "fcd_filt_3",
        "fcd_filt_4",
        "fcd_filt_5",
        "fcd_filt_6",
        "fcd_filt_7",
        'fcd_corr_0',
        'fcd_corr_1',
        'fcd_corr_2',
        'fcd_corr_3',
        'fcd_corr_4',
        'fcd_corr_5',
        'fcd_corr_6',
        'fcd_corr_7',
        'fc_corr_0',
        'fc_corr_1',
        'fc_corr_2',
        'fc_corr_3',
        'fc_corr_4',
        'fc_corr_5',
        'fc_corr_6',
        'fc_corr_7'
    ]

    features_need_psd = ["spectral_power",
                         "max_PSD_freq",
                         "PSD_under_area",
                         "PSD_under_power"]
    features_need_fcd = ["fcd_corr",
                         "fcd_elements",
                         "fcd_sum",
                         "fluidity"]
    # -------------------------------------------------------------------------

    def __init__(self, features=[], opts=None) -> None:

        self.features = features
        for feature in features:
            # if feature.startswith("fcd_mask"):
            #     continue
            if not self.feature_is_valid(feature):
                raise ValueError("Invalid feature")

        self.set_default_params()

        if not opts is None:
            self.set_feature_params(opts)
    # -------------------------------------------------------------------------

    def feature_is_valid(self, feature):
        '''!
        check if feature is valid

        Parameters
        ----------
        feature: str
            feature name

        Returns
        -------
        bool
            True if feature is valid
        '''

        return feature in self.available_features
    # -------------------------------------------------------------------------

    def set_default_params(self):
        '''
        Set default parameters for feature calculation
        '''

        opt_FCD_common = {"k": 30, "PCA_n_components": 3, "wwidth": 30, "maxNwindows": 200,
                          "olap": 0.94, "mode": "corr", "axis": 1, 'verbose': False, 'demean': False}
        opt_PSD_common = {'axis': 1,  # !TODO check for axis=0
                          'fs': None,
                          'method': 'periodogram',  # 'welch'
                          'nperseg': 2048,
                          'noverlap': 0,
                          'average_over_channels': False,
                          'regions': [],
                          }

        self.opt_fcd_sum = deepcopy(opt_FCD_common)
        self.opt_fcd_corr = deepcopy(opt_FCD_common)
        self.opt_FCD_common = deepcopy(opt_FCD_common)
        self.opt_fcd_elements = deepcopy(opt_FCD_common)
        self.opt_fcd_filt = {"k": 1, "win_sp": 1, "win_len": 20, "verbose": False, "regions": None}
        self.opt_fcd_filt_limbic = {"k": 1, "win_sp": 1, "win_len": 20, "verbose": False, "regions": None}
        self.opt_fcd_filt_temporal = {"k": 1, "win_sp": 1, "win_len": 20, "verbose": False, "regions": None}
        self.opt_fcd_filt_frontal = {"k": 1, "win_sp": 1, "win_len": 20, "verbose": False, "regions": None}
        self.opt_fcd_filt_occipital = {"k": 1, "win_sp": 1, "win_len": 20, "verbose": False, "regions": None}
        self.opt_fcd_filt_parietal = {"k": 1, "win_sp": 1, "win_len": 20, "verbose": False, "regions": None}
        self.opt_fcd_filt_centralstructures = {"k": 1, "win_sp": 1, "win_len": 20, "verbose": False, "regions": None}
        self.opt_fcd_filt_0 = {"k": 1, "win_sp": 1, "win_len": 20, "verbose": False, "regions": None}
        self.opt_fcd_filt_1 = {"k": 1, "win_sp": 1, "win_len": 20, "verbose": False, "regions": None}
        self.opt_fcd_filt_2 = {"k": 1, "win_sp": 1, "win_len": 20, "verbose": False, "regions": None}
        self.opt_fcd_filt_3 = {"k": 1, "win_sp": 1, "win_len": 20, "verbose": False, "regions": None}
        self.opt_fcd_filt_4 = {"k": 1, "win_sp": 1, "win_len": 20, "verbose": False, "regions": None}
        self.opt_fcd_filt_5 = {"k": 1, "win_sp": 1, "win_len": 20, "verbose": False, "regions": None}
        self.opt_fcd_filt_6 = {"k": 1, "win_sp": 1, "win_len": 20, "verbose": False, "regions": None}
        self.opt_fcd_filt_7 = {"k": 1, "win_sp": 1, "win_len": 20, "verbose": False, "regions": None}
        self.opt_fcd_corr_0 = {"k": 1, "win_sp": 1, "win_len": 20, "verbose": False, "regions": None}
        self.opt_fcd_corr_1 = {"k": 1, "win_sp": 1, "win_len": 20, "verbose": False, "regions": None}
        self.opt_fcd_corr_2 = {"k": 1, "win_sp": 1, "win_len": 20, "verbose": False, "regions": None}
        self.opt_fcd_corr_3 = {"k": 1, "win_sp": 1, "win_len": 20, "verbose": False, "regions": None}
        self.opt_fcd_corr_4 = {"k": 1, "win_sp": 1, "win_len": 20, "verbose": False, "regions": None}
        self.opt_fcd_corr_5 = {"k": 1, "win_sp": 1, "win_len": 20, "verbose": False, "regions": None}
        self.opt_fcd_corr_6 = {"k": 1, "win_sp": 1, "win_len": 20, "verbose": False, "regions": None}
        self.opt_fcd_corr_7 = {"k": 1, "win_sp": 1, "win_len": 20, "verbose": False, "regions": None}
        self.opt_fc_corr_0 = {'k': 1, 'PCA_n_components': 3, 'demean': False, 'regions': None}
        self.opt_fc_corr_1 = {'k': 1, 'PCA_n_components': 3, 'demean': False, 'regions': None}
        self.opt_fc_corr_2 = {'k': 1, 'PCA_n_components': 3, 'demean': False, 'regions': None}
        self.opt_fc_corr_3 = {'k': 1, 'PCA_n_components': 3, 'demean': False, 'regions': None}
        self.opt_fc_corr_4 = {'k': 1, 'PCA_n_components': 3, 'demean': False, 'regions': None}
        self.opt_fc_corr_5 = {'k': 1, 'PCA_n_components': 3, 'demean': False, 'regions': None}
        self.opt_fc_corr_6 = {'k': 1, 'PCA_n_components': 3, 'demean': False, 'regions': None}
        self.opt_fc_corr_7 = {'k': 1, 'PCA_n_components': 3, 'demean': False, 'regions': None}

        self.opt_phFCD = {"k": 1, 'PCA_n_components': 3}

        # opt_PSD_under_area = {
        #     "bands": ["delta", "theta", "alpha", "beta", "gamma", "high_gamma"]}

        self.opt_fcd_regions = deepcopy(opt_FCD_common)
        self.opt_fcd_regions.update({'regions': []})
        self.opt_fluidity = {"k": 1}

        self.opt_fc_sum = {}
        self.opt_fc_corr = {'k': 10, "PCA_n_components": 3, 'demean': False}
        self.opt_fc_homotopic = {'positive': True, "avg": False}
        self.opt_fc_corr_regions = {'k': 10,
                                    'regions': [],
                                    'PCA_n_components': 3}

        self.opt_burstiness = {'axis': 1}
        self.opt_custom_skewness = {'axis': 1, "whatSkew": "pearson"}

        self.opt_PSD_common = deepcopy(opt_PSD_common)
        self.opt_max_PSD_freq = deepcopy(opt_PSD_common)
        self.opt_max_PSD_freq.update({"normalize": False})

        self.opt_PSD_under_area = deepcopy(opt_PSD_common)
        self.opt_PSD_under_area.update({"fmin": 1.0,
                                        "fmax": 50.0,
                                        "normalize": False,
                                        "average_over_channels": False})
        # !TODO check fs update to extract PSD

        self.opt_PSD_under_power = deepcopy(opt_PSD_common)
        self.opt_PSD_under_power.update({"fmin": 1.0, "fmax": 50.0})

        self.opt_spectral_power = deepcopy(opt_PSD_common)

        self.opt_spectral_connectivity = {'method': 'plv', "decimate": 1,
                                          "k": 10, "PCA_n_components": 10,
                                          'filt_band': False,  # !BUG if True return nan
                                          'fmin': 0.0,
                                          'fmax': 100.0,
                                          'fs': None}

        self.opt_moments = {'axis': 1, "regions": [], "demean": False}
        self.opt_high_low_mu = {'axis': 1}
        self.opt_IQR = {'axis': 1}
        self.opt_abs_moments = {}
        self.opt_higher_moments = {'axis': 1, "regions": []}
        self.opt_envelope = {'axis': 1, "regions": []}
        self.opt_fcd_edge = {'k': 30, "PCA_n_components": 3}
        self.opt_fcd_edge_var = {}
        self.opt_KOP = {"hilbert": True}
        self.opt_local_to_global_coherence = {'regions': []}

        self.opt_preprocessing = {'bandpass': None, 
                                  'detrend': False,  
                                  'demean': False,           
                                  'zscore': False,  
                                  'removeStrongArtefacts': False,         
                                  'offset': 0}
        self.opt_coactivation = {}
        self.opt_coactivation_ph = {}

    # -------------------------------------------------------------------------

    def set_feature_params(self, opts):
        '''
        Set parameters for feature calculation

        Parameters
        ----------
        opt : dictionary of dictionaries
            containing parameters for each feature calculation
        '''

        for feature in opts.keys():
            assert(self.feature_is_valid(feature))
            attribute = "opt_" + feature

            # store new dictionary in temporary variable and then update the default one
            tmp = getattr(self, attribute)
            assert(tmp is not None), "Feature {} not available".format(attribute)
            tmp.update(opts[feature])
            setattr(self, attribute, tmp)

            if feature in self.features_need_psd:
                tmp = getattr(self, "opt_PSD_common")
                assert(tmp is not None), "PSD common parameters are not set"
                tmp.update(opts[feature])
                setattr(self, "opt_PSD_common", tmp)
                # update also the feature specific parameters
                tmp = getattr(self, attribute)
                assert(tmp is not None), "Feature {} not available".format(
                    attribute)
                tmp.update(opts[feature])
                setattr(self, attribute, tmp)

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    def calc_features(self, data):
        '''!
        extract features from given data

        Parameters
        ----------
        data: np.ndarray (2d)
            input array
        opt: dictiionary of dictionaries
            dictionary of dictionaries containing parameters for each feature calculation
        Returns
        -------
        stats_vec: np.ndarray (1d)
            feature vector
        stats_info: dict
            dictionary containing information about the features

        '''

        features = self.features

        if "preprocessing" in features:
            data = preprocessing_signal(data, self.opt_preprocessing)

        def get_length(x):
            return (len(x)) if (len(x) > 0) else 0

        stats_vec = np.array([])
        stats_info = {}

        featuers_need_fcd = self.features_need_fcd
        features_need_psd = self.features_need_psd

        if any([f in features for f in featuers_need_fcd]):
            FCD_res = self.extract_FCD(data, self.opt_FCD_common)

        if any([f in features for f in features_need_psd]):
            freq, Pxx = self.extract_PSD(data, self.opt_PSD_common)
            # if self.opt_PSD_common['average_over_channels']:
            #     Pxx = np.mean(Pxx, axis=0).reshape(1, -1)

        for feature in features:

            if feature == "preprocessing":
                continue

            c = get_length(stats_vec)
            func_feature = getattr(self, feature)
            opt_name = "opt_" + feature
            opt_feature = getattr(self, opt_name)

            # FCD features
            if feature in featuers_need_fcd:
                stats_vec = np.append(
                    stats_vec, func_feature(FCD_res, opt_feature))

            # PSD features
            elif feature in features_need_psd:
                stats_vec = np.append(
                    stats_vec, func_feature(freq, Pxx, opt_feature))
            # elif feature == "fcd_mask":
            #     func_feature = getattr(self, "fcd_mask")
            #     stats_vec = np.append(
            #         stats_vec, func_feature(data, opt_feature))

            else:
                stats_vec = np.append(
                    stats_vec, func_feature(data, opt_feature))
            stats_info[feature] = [c, get_length(stats_vec)]

        return stats_vec, stats_info
    # -------------------------------------------------------------------------

    # def fcd_mask(self, x, opt=None):
    #     '''
    #     extract feature from fcd mask
    #     '''
    #     if opt is None:
    #         opt = self.opt_fcd_mask
        
    #     FCD_res = self.get_fcd_mask(x, opt)
    #     k = opt['k']
    #     PCA_n_components = opt['PCA_n_components']

    #     if len(fcd_red) == 1:  # find nan value
    #         return FCD_res
    #     else:
    #         FCDcorr = FCD_res
    #         off_diag_sum_FCD = np.sum(
    #             np.abs(FCDcorr)) - np.trace(np.abs(FCDcorr))
    #         off_diag_sum_FCD = np.sum(set_k_diogonal(FCDcorr, k, 0.0))

    #         FCD_TRIU = np.triu(FCDcorr, k=k)

    #         eigen_vals_FCD, _ = LA.eig(FCDcorr)
    #         pca = PCA(n_components=PCA_n_components)
    #         PCA_FCD = pca.fit_transform(FCDcorr)

    #         Upper_FCD = []
    #         Lower_FCD = []
    #         for i in range(0, len(FCDcorr)):
    #             Upper_FCD.extend(FCDcorr[i][i+1:])
    #             Lower_FCD.extend(FCDcorr[i][0:i])

    #         stats_vec = np.array([])
    #         if not demean:
    #             funcs = [np.sum, np.max, np.min, np.mean,
    #                      np.std, skew, kurtosis]
    #         else:
    #             funcs = [np.std, skew, kurtosis]
    #         data = [eigen_vals_FCD.reshape(-1),
    #                 PCA_FCD.reshape(-1),
    #                 Upper_FCD,
    #                 Lower_FCD,
    #                 FCD_TRIU.reshape(-1)]

    #         for ki in range(len(data)):
    #             _st = np.zeros(len(funcs))
    #             for i in range(len(funcs)):
    #                 _st[i] = funcs[i](data[ki])
    #             stats_vec = np.append(stats_vec, _st)

    #         stats_vec = np.append(stats_vec, np.quantile(
    #             FCDcorr, [0.05, 0.25, 0.5, 0.75, 0.95]))
    #         stats_vec = np.append(stats_vec, [off_diag_sum_FCD])

    #         return stats_vec


    # def get_fcd_mask(self, x, opt=None):
    #     '''
    #     return fcd mask
    #     '''
    #     if opt is None:
    #         opt = self.opt_fcd_mask

    #     regions_idx = opt['regions']

    #     nn, nt = x.shape
    #     maskregions = np.zeros((nn, nn))
    #     maskregions[np.ix_(regions_idx, regions_idx)] = 1

    #     result = self.compute_fcd_regions_filt(x.T,
    #                                            maskregions,
    #                                            win_len=opt['win_len'],
    #                                            win_sp=opt['win_sp'],
    #                                            verbose=opt['verbose'],)

    #     if len(result) == 1:
    #         if verbose:
    #             print(np.isnan(result).any())
    #         return result
    #     else:
    #         return result[0]

    def fc_sum(self, x, opt=None):
        '''!
        Calculate the sum of functional connectivity (FC)

        x: np.ndarray (2d)
           input BOLD signal [nnodes, ntime]

        \returns np.ndarray (1d)
            sum of functional connectivity
        '''
        FCcorr = np.corrcoef(x)
        off_diag_sum_FC = np.sum(np.abs(FCcorr)) - np.trace(np.abs(FCcorr))

        return off_diag_sum_FC

    def fc_elements(self, x, opt=None):
        '''!
        Calculate the  functional connectivity (FC)

        \param x np.ndarray (2d)
            input BOLD signal [nnodes, ntime]
        \returns np.ndarray (1d)
            functional connectivity elements
        '''
        return np.corrcoef(x).reshape(-1)

    def fc_corr(self, x, opt=None):
        '''!
        calculate the freatures from functional connectivity (FC)

        Parameters
        ----------
        x: np.ndarray (2d)
            input array
        opt: dict
            dictionary of parameters
        Returns
        -------
        stats: np.ndarray (1d)
            feature values
        '''
        if opt is None:
            opt = self.opt_fc_corr
        demean = opt["demean"]

        def funcs(x, demean=False):
            if demean:
                vec = np.zeros(3)
                vec[0] = np.std(x)
                vec[1] = skew(x)
                vec[2] = kurtosis(x)

            else:
                vec = np.zeros(7)
                vec[0] = np.sum(x)
                vec[1] = np.max(x)
                vec[2] = np.min(x)
                vec[3] = np.mean(x)
                vec[4] = np.std(x)
                vec[5] = skew(x)
                vec[6] = kurtosis(x)
            return vec

        FCcorr = np.corrcoef(x)
        off_diag_sum_FC = np.sum(np.abs(FCcorr)) - np.trace(np.abs(FCcorr))

        FC_TRIU = np.triu(FCcorr, k=opt["k"])
        eigen_vals_FC, _ = LA.eig(FCcorr)
        pca = PCA(n_components=opt["PCA_n_components"])
        PCA_FC = pca.fit_transform(FCcorr)

        Upper_FC = []
        Lower_FC = []
        for i in range(0, len(FCcorr)):
            Upper_FC.extend(FCcorr[i][i+1:])
            Lower_FC.extend(FCcorr[i][0:i])

        q = np.quantile(FCcorr, [0.05, 0.25, 0.5, 0.75, 0.95])
        _stats = np.array([])
        _stats = np.append(_stats, q)
        _stats = np.append(_stats, funcs(Upper_FC, demean))
        _stats = np.append(_stats, funcs(Lower_FC, demean))
        _stats = np.append(_stats, funcs(PCA_FC.reshape(-1), demean))
        _stats = np.append(_stats, funcs(FC_TRIU.reshape(-1), demean))
        _stats = np.append(_stats, funcs(np.real(eigen_vals_FC[:-1]), demean))

        # keep this the last element
        _stats = np.append(_stats, [off_diag_sum_FC])

        return _stats
    # -------------------------------------------------------------------------

    def fc_corr_0(self, x, opt=None):
        if opt is None:
            opt = self.opt_fc_corr_0
        regions = opt["regions"]
        fc = fc_mask(x, regions,  opt)
        return mat_stats(fc, opt)
    
    def fc_corr_1(self, x, opt=None):
        if opt is None:
            opt = self.opt_fc_corr_1
        regions = opt["regions"]
        fc = fc_mask(x, regions,  opt)
        return mat_stats(fc, opt)
    
    def fc_corr_2(self, x, opt=None):
        if opt is None:
            opt = self.opt_fc_corr_2
        regions = opt["regions"]
        fc = fc_mask(x, regions,  opt)
        return mat_stats(fc, opt)

    def fc_corr_3(self, x, opt=None):
        if opt is None:
            opt = self.opt_fc_corr_3
        regions = opt["regions"]
        fc = fc_mask(x, regions,  opt)
        return mat_stats(fc, opt)

    def fc_corr_4(self, x, opt=None):
        if opt is None:
            opt = self.opt_fc_corr_4
        regions = opt["regions"]
        fc = fc_mask(x, regions,  opt)
        return mat_stats(fc, opt)

    def fc_corr_5(self, x, opt=None):
        if opt is None:
            opt = self.opt_fc_corr_5
        regions = opt["regions"]
        fc = fc_mask(x, regions,  opt)
        return mat_stats(fc, opt)

    def fc_corr_6(self, x, opt=None):
        if opt is None:
            opt = self.opt_fc_corr_6
        regions = opt["regions"]
        fc = fc_mask(x, regions,  opt)
        return mat_stats(fc, opt)

    def fc_corr_7(self, x, opt=None):
        if opt is None:
            opt = self.opt_fc_corr_7
        regions = opt["regions"]
        fc = fc_mask(x, regions,  opt)
        return mat_stats(fc, opt)


    def fc_corr_regions(self, x, opt=None):
        '''!
        calculate the freatures from functional connectivity (FC) on given regions

        \param x np.ndarray (2d)
            input array
        \param regions  list[int]
            index of regions to use for FC
        \return np.ndarray (1d)
            list of feature values
        '''

        if opt is None:
            opt = self.opt_fc_corr_regions

        k = opt["k"]
        regions = opt["regions"]
        PCA_n_components = opt["PCA_n_components"]

        assert(len(regions) >=
               2), 'regions should be at least 2, use set_feature_properties to set regions'
        assert(isinstance(regions[0], (np.int64, int, np.int32)))

        def funcs(x):
            vec = np.zeros(7)
            vec[0] = np.sum(x)
            vec[1] = np.max(x)
            vec[2] = np.min(x)
            vec[3] = np.mean(x)
            vec[4] = np.std(x)
            vec[5] = skew(x)
            vec[6] = kurtosis(x)
            return vec

        x_ = copy(x[regions, :])

        rsFC = np.corrcoef(x_)

        #! Added based on Fox et al., 2009; Murphy et al., 2009; Murphy and Fox, 2017
        # rsFC = rsFC * (rsFC > 0.0)
        off_diag_sum_FC = np.sum(np.abs(rsFC)) - np.trace(np.abs(rsFC))

        FC_TRIU = np.triu(rsFC, k=k)
        eigen_vals_FC, _ = LA.eig(rsFC)
        pca = PCA(n_components=PCA_n_components)
        PCA_FC = pca.fit_transform(rsFC)

        Upper_FC = []
        Lower_FC = []
        for i in range(0, len(rsFC)):
            Upper_FC.extend(rsFC[i][i+1:])
            Lower_FC.extend(rsFC[i][0:i])

        q = np.quantile(rsFC, [0.25, 0.5, 0.75])

        _stats = np.array([])
        _stats = np.append(_stats, q)
        _stats = np.append(_stats, funcs(Upper_FC))
        _stats = np.append(_stats, funcs(Lower_FC))
        _stats = np.append(_stats, funcs(PCA_FC.reshape(-1)))
        _stats = np.append(_stats, funcs(FC_TRIU.reshape(-1)))
        # last element produce core dump error in the training
        _stats = np.append(_stats, funcs(np.real(eigen_vals_FC[:-1])))

        # keep this the last element
        _stats = np.append(_stats, [off_diag_sum_FC])

        return _stats
    # -------------------------------------------------------------------------

    def burstiness(self, x, opt=None):
        '''!
        calculate the burstiness statistic from
        [from hctsa-py]

        \param y the input time series
        \returns np.ndarray (1d)
            The burstiness statistic, B.

        - Goh and Barabasi, 'Burstiness and memory in complex systems' Europhys. Lett.
        81, 48002 (2008).
        '''

        if opt is None:
            opt = self.opt_burstiness

        if x.mean() == 0:
            return np.nan
        axis = opt['axis']
        r = np.std(x, axis=axis) / np.mean(x, axis=axis)
        B = (r - 1) / (r + 1)

        return B
    # -------------------------------------------------------------------------

    def custom_skewness(self, y, opt=None):
        '''!
        Calculate the skewness of the given 2d signal (BOLD)
        from hctsa-py

        Parameters
        ----------
        y: np.ndarray (2d)
            input BOLD signal [nnodes, ntime]
        opt: dict
            parameters
            axis: int
                axis along which to compute the skewness
            whatSkew: str
                options: 'pearson', 'bowley'

        Returns
        -------
        skewness: np.ndarray (1d)
            skewness of the signal
        '''

        if opt is None:
            opt = self.opt_custom_skewness

        axis = opt['axis']
        whatSkew = opt['whatSkew']

        if whatSkew == 'pearson':
            if np.std(y) != 0:
                return (3*np.mean(y, axis=axis) - np.median(y, axis=axis)) / np.std(y, axis=axis)
            else:
                return 0
        elif whatSkew == 'bowley':
            qs = np.quantile(y, [.25, .5, .75], axis=axis)
            if np.std(y) != 0:
                return (qs[2] + qs[0] - 2*qs[1]) / (qs[2] - qs[0])
            else:
                return 0
        else:
            raise Exception('whatSkew must be either pearson or bowley.')
    # -------------------------------------------------------------------------

    def fc_homotopic(self, bold, opt=None):
        '''!
        Calculate the homotopic connectivity vector of a given brain activity

        Parameters
        ----------
        bold: array_like [n_nodes, n_samples]
            The brain activity to be analyzed.
        Returns
        -------
        Homotopic_FC_vector : array_like [n_nodes]
            The homotopic correlation vector.

        Negative correlations may be artificially induced when using global signal regression
        in functional imaging pre-processing (Fox et al., 2009; Murphy et al., 2009; Murphy and Fox, 2017).
        Therefore, results on negative weights should be interpreted with caution and should be understood
        as complementary information underpinning the findings based on positive connections
        '''
        if opt is None:
            opt = self.opt_fc_homotopic

        positive = opt['positive']
        avg = opt['avg']

        NHALF = int(bold.shape[0]//2)
        rsFC = np.corrcoef(bold)
        if positive:
            rsFC = rsFC * (rsFC > 0)
        rsFC = rsFC - np.diag(np.diag(rsFC))
        Homotopic_FC = np.diag(rsFC, k=NHALF)
        if avg:
            return [np.mean(Homotopic_FC)]
        else:
            return Homotopic_FC
    # -------------------------------------------------------------------------

    def spectral_connectivity(self, x, opt=None):
        '''!
        calculate the freatures from phase locking value (PLV) matrix

        Parameters
        ----------
        x: np.ndarray (2d)
            input array
        opt: dict
            dictionary of parameters
        Returns
        -------
        stats: np.ndarray (1d)
            feature values
        '''

        if opt is None:
            opt = self.opt_spectral_connectivity

        k = opt['k']
        fs = opt['fs']
        fmin = opt['fmin']
        fmax = opt['fmax']
        method = opt['method']
        decimate = opt['decimate']
        filt_band = opt['filt_band']
        PCA_n_components = opt['PCA_n_components']

        assert (not fs is None), 'fs (frequency sampling) must be provided'

        def funcs(x):
            vec = np.zeros(7)
            vec[0] = np.sum(x)
            vec[1] = np.max(x)
            vec[2] = np.min(x)
            vec[3] = np.mean(x)
            vec[4] = np.std(x)
            vec[5] = skew(x)
            vec[6] = kurtosis(x)
            return vec

        if filt_band:
            x = filter_butter_bandpass(x, fs, fmin, fmax, order=5)

        if method == 'plv':
            mat = compute_plv(x[:, ::decimate])
        elif method == 'pli':
            mat = compute_pli(x[:, ::decimate])
        else:
            raise Exception('method must be either plv or pli')
        off_diag_sum = np.sum(np.abs(mat)) - np.trace(np.abs(mat))

        # _triu = np.triu(mat, k=k)
        _eigen_vals, _ = LA.eig(mat)
        pca = PCA(n_components=PCA_n_components)
        _PCA = pca.fit_transform(mat)

        _Upper_FC = []
        _Lower_FC = []
        for i in range(0, len(mat)):
            _Upper_FC.extend(mat[i][i+1:])
            _Lower_FC.extend(mat[i][0:i])

        q = np.quantile(mat, [0.25, 0.5, 0.75])

        _stats = np.array([])
        _stats = np.append(_stats, q)
        _stats = np.append(_stats, funcs(_Upper_FC))
        _stats = np.append(_stats, funcs(_Lower_FC))
        _stats = np.append(_stats, funcs(_PCA.reshape(-1)))
        # _stats = np.append(_stats, funcs(_triu.reshape(-1))) #! dump
        _stats = np.append(_stats, funcs(np.real(_eigen_vals[:-1])))

        # keep this the last element
        _stats = np.append(_stats, [off_diag_sum])

        return _stats
    # -------------------------------------------------------------------------

    def fluidity(self, FCD_res, opt=None):
        '''!
        Calculate the fluidity of the BOLD signal

        \param FCD np.ndarray (2d)
            input FCD matrix
        \param k_diagonal int
            number of subdiagonal to be excluded from FCD matrix
        \returns np.ndarray (1d)
            list of feature values
        '''

        if opt is None:
            opt = self.opt_fluidity

        k = opt['k']

        if len(FCD_res) == 1:  # find nan value
            return FCD_res
        else:
            FCDcorr, _, _ = FCD_res
            # FCDcorr = select_upper_triangular(FCDcorr, k)
            FCDcorr = np.triu(FCDcorr, k=k)
            return np.var(FCDcorr.reshape(-1))

    # -------------------------------------------------------------------------

    def fcd_filt(self, x, opt=None):
        '''!
        Calculate the functional connectivity dynamics (FCD) of the BOLD signal
        for given regions.

        Returns
        -------

        x: list of 2 floats
            mean and variance of FCD matrix for given regions.
        '''

        if opt is None:
            opt = self.opt_fcd_filt

        win_len = opt['win_len']
        win_sp = opt['win_sp']
        regions = opt['regions']
        assert(regions is not None), 'regions_idx must be provided in features_opt'
        verbose = opt['verbose']
        k = opt['k']

        return _fcd_filt(x, regions, win_len, win_sp, k, verbose)

    def fcd_filt_limbic(self, x, opt=None):

        if opt is None:
            opt = self.opt_fcd_filt_limbic

        win_len = opt['win_len']
        win_sp = opt['win_sp']
        regions = opt['regions']
        assert(regions is not None), 'regions_idx must be provided in features_opt'
        verbose = opt['verbose']
        k = opt['k']

        return _fcd_filt(x, regions, win_len, win_sp, k, verbose)

    def fcd_filt_frontal(self, x, opt=None):

        if opt is None:
            opt = self.opt_fcd_filt_frontal

        win_len = opt['win_len']
        win_sp = opt['win_sp']
        regions = opt['regions']
        assert(regions is not None), 'regions_idx must be provided in features_opt'
        verbose = opt['verbose']
        k = opt['k']

        return _fcd_filt(x, regions, win_len, win_sp, k, verbose)

    def fcd_filt_temporal(self, x, opt=None):

        if opt is None:
            opt = self.opt_fcd_filt_temporal
            
        win_len = opt['win_len']
        win_sp = opt['win_sp']
        regions = opt['regions']
        assert(regions is not None), 'regions_idx must be provided in features_opt'
        verbose = opt['verbose']
        k = opt['k']

        return _fcd_filt(x, regions, win_len, win_sp, k, verbose)

    def fcd_filt_occipital(self, x, opt=None):

        if opt is None:
            opt = self.opt_fcd_filt_occipital
            
        win_len = opt['win_len']
        win_sp = opt['win_sp']
        regions = opt['regions']
        assert(regions is not None), 'regions_idx must be provided in features_opt'
        verbose = opt['verbose']
        k = opt['k']

        return _fcd_filt(x, regions, win_len, win_sp, k, verbose)

    def fcd_filt_centralstructures(self, x, opt=None):

        if opt is None:
            opt = self.opt_fcd_filt_centralstructures
                
        win_len = opt['win_len']
        win_sp = opt['win_sp']
        regions = opt['regions']
        assert(regions is not None), 'regions_idx must be provided in features_opt'
        verbose = opt['verbose']
        k = opt['k']

        return _fcd_filt(x, regions, win_len, win_sp, k, verbose)
    
    def fcd_filt_parietal(self, x, opt=None):

        if opt is None:
            opt = self.opt_fcd_filt_parietal
        
        win_len = opt['win_len']
        win_sp = opt['win_sp']
        regions = opt['regions']
        assert(regions is not None), 'regions_idx must be provided in features_opt'
        verbose = opt['verbose']
        k = opt['k']

        return _fcd_filt(x, regions, win_len, win_sp, k, verbose)
    
    def fcd_corr_0(self, x, opt=None):
        if opt is None:
            opt = self.opt_fcd_corr_0
        win_len = opt['win_len']
        win_sp = opt['win_sp']
        regions = opt['regions']
        assert(regions is not None), 'regions_idx must be provided in features_opt'
        verbose = opt['verbose']
        k = opt['k']
        fcd_ = fcd_mask(x, regions, win_len=win_len, win_sp=win_sp, verbose=verbose)
        if not np.isnan(fcd_).any():
            return self.fcd_corr([fcd_, 0, 0], opt={'k': k, 'PCA_n_components': 3, 'demean': False})
        else:
            return [np.nan]
    
    def fcd_corr_1(self, x, opt=None):
        if opt is None:
            opt = self.opt_fcd_corr_1
        win_len = opt['win_len']
        win_sp = opt['win_sp']
        regions = opt['regions']
        assert(regions is not None), 'regions_idx must be provided in features_opt'
        verbose = opt['verbose']
        k = opt['k']
        fcd_ = fcd_mask(x, regions, win_len=win_len, win_sp=win_sp, verbose=verbose)
        if not np.isnan(fcd_).any():
            return self.fcd_corr([fcd_, 0, 0], opt={'k': k, 'PCA_n_components': 3, 'demean': False})
        else:
            return [np.nan]
    
    def fcd_corr_2(self, x, opt):
        if opt is None:
            opt = self.opt_fcd_corr_2
        win_len = opt['win_len']
        win_sp = opt['win_sp']
        regions = opt['regions']
        assert(regions is not None), 'regions_idx must be provided in features_opt'
        verbose = opt['verbose']
        k = opt['k']
        fcd_ = fcd_mask(x, regions, win_len=win_len, win_sp=win_sp, verbose=verbose)
        if not np.isnan(fcd_).any():
            return self.fcd_corr([fcd_, 0, 0], opt={'k': k, 'PCA_n_components': 3, 'demean': False})
        else:
            return [np.nan]
    
    def fcd_corr_3(self, x, opt=None):
        if opt is None:
            opt = self.opt_fcd_corr_3
        win_len = opt['win_len']
        win_sp = opt['win_sp']
        regions = opt['regions']
        assert(regions is not None), 'regions_idx must be provided in features_opt'
        verbose = opt['verbose']
        k = opt['k']
        fcd_ = fcd_mask(x, regions, win_len=win_len, win_sp=win_sp, verbose=verbose)
        if not np.isnan(fcd_).any():
            return self.fcd_corr([fcd_, 0, 0], opt={'k': k, 'PCA_n_components': 3, 'demean': False})
        else:
            return [np.nan]
    
    def fcd_corr_4(self, x, opt=None):
        if opt is None:
            opt = self.opt_fcd_corr_4
        win_len = opt['win_len']
        win_sp = opt['win_sp']
        regions = opt['regions']
        assert(regions is not None), 'regions_idx must be provided in features_opt'
        verbose = opt['verbose']
        k = opt['k']
        fcd_ = fcd_mask(x, regions, win_len=win_len, win_sp=win_sp, verbose=verbose)
        if not np.isnan(fcd_).any():
            return self.fcd_corr([fcd_, 0, 0], opt={'k': k, 'PCA_n_components': 3, 'demean': False})
        else:
            return [np.nan]

    def fcd_corr_5(self, x, opt=None):
        if opt is None:
            opt = self.opt_fcd_corr_5
        win_len = opt['win_len']
        win_sp = opt['win_sp']
        regions = opt['regions']
        assert(regions is not None), 'regions_idx must be provided in features_opt'
        verbose = opt['verbose']
        k = opt['k']
        fcd_ = fcd_mask(x, regions, win_len=win_len, win_sp=win_sp, verbose=verbose)
        if not np.isnan(fcd_).any():
            return self.fcd_corr([fcd_, 0, 0], opt={'k': k, 'PCA_n_components': 3, 'demean': False})
        else:
            return [np.nan]
    
    def fcd_corr_6(self, x, opt=None):
        if opt is None:
            opt = self.opt_fcd_corr_6
        win_len = opt['win_len']
        win_sp = opt['win_sp']
        regions = opt['regions']
        assert(regions is not None), 'regions_idx must be provided in features_opt'
        verbose = opt['verbose']
        k = opt['k']
        fcd_ = fcd_mask(x, regions, win_len=win_len, win_sp=win_sp, verbose=verbose)
        if not np.isnan(fcd_).any():
            return self.fcd_corr([fcd_, 0, 0], opt={'k': k, 'PCA_n_components': 3, 'demean': False})
        else:
            return [np.nan]
    
    def fcd_corr_7(self, x, opt=None):
        if opt is None:
            opt = self.opt_fcd_corr_7
        win_len = opt['win_len']
        win_sp = opt['win_sp']
        regions = opt['regions']
        assert(regions is not None), 'regions_idx must be provided in features_opt'
        verbose = opt['verbose']
        k = opt['k']
        fcd_ = fcd_mask(x, regions, win_len=win_len, win_sp=win_sp, verbose=verbose)
        if not np.isnan(fcd_).any():
            return self.fcd_corr([fcd_, 0, 0], opt={'k': k, 'PCA_n_components': 3, 'demean': False})
        else:
            return [np.nan]

    def fcd_filt_0(self, x, opt=None):
        if opt is None:
            opt = self.opt_fcd_filt_0
        win_len = opt['win_len']
        win_sp = opt['win_sp']
        regions = opt['regions']
        assert(regions is not None), 'regions_idx must be provided in features_opt'
        verbose = opt['verbose']
        k = opt['k']

        return _fcd_filt(x, regions, win_len, win_sp, k, verbose)
    
    def fcd_filt_1(self, x, opt=None):
        if opt is None:
            opt = self.opt_fcd_filt_1
                
        win_len = opt['win_len']
        win_sp = opt['win_sp']
        regions = opt['regions']
        assert(regions is not None), 'regions_idx must be provided in features_opt'
        verbose = opt['verbose']
        k = opt['k']

        return _fcd_filt(x, regions, win_len, win_sp, k, verbose)

    def fcd_filt_2(self, x, opt=None):
        if opt is None:
            opt = self.opt_fcd_filt_2
        win_len = opt['win_len']
        win_sp = opt['win_sp']
        regions = opt['regions']
        assert(regions is not None), 'regions_idx must be provided in features_opt'
        verbose = opt['verbose']
        k = opt['k']

        return _fcd_filt(x, regions, win_len, win_sp, k, verbose)
    
    def fcd_filt_3(self, x, opt=None):
        if opt is None:
            opt = self.opt_fcd_filt_3
        win_len = opt['win_len']
        win_sp = opt['win_sp']
        regions = opt['regions']
        assert(regions is not None), 'regions_idx must be provided in features_opt'
        verbose = opt['verbose']
        k = opt['k']

        return _fcd_filt(x, regions, win_len, win_sp, k, verbose)
    
    def fcd_filt_4(self, x, opt=None):
        if opt is None:
            opt = self.opt_fcd_filt_4
        win_len = opt['win_len']
        win_sp = opt['win_sp']
        regions = opt['regions']
        assert(regions is not None), 'regions_idx must be provided in features_opt'
        verbose = opt['verbose']
        k = opt['k']

        return _fcd_filt(x, regions, win_len, win_sp, k, verbose)
    
    def fcd_filt_5(self, x, opt=None):
        if opt is None:
            opt = self.opt_fcd_filt_5
        win_len = opt['win_len']
        win_sp = opt['win_sp']
        regions = opt['regions']
        assert(regions is not None), 'regions_idx must be provided in features_opt'
        verbose = opt['verbose']
        k = opt['k']

        return _fcd_filt(x, regions, win_len, win_sp, k, verbose)
    
    def fcd_filt_6(self, x, opt=None):
        if opt is None:
            opt = self.opt_fcd_filt_6
        win_len = opt['win_len']
        win_sp = opt['win_sp']
        regions = opt['regions']
        assert(regions is not None), 'regions_idx must be provided in features_opt'
        verbose = opt['verbose']
        k = opt['k']

        return _fcd_filt(x, regions, win_len, win_sp, k, verbose)
    
    def fcd_filt_7(self, x, opt=None):
        if opt is None:
            opt = self.opt_fcd_filt_7
        win_len = opt['win_len']
        win_sp = opt['win_sp']
        regions = opt['regions']
        assert(regions is not None), 'regions_idx must be provided in features_opt'
        verbose = opt['verbose']
        k = opt['k']

        return _fcd_filt(x, regions, win_len, win_sp, k, verbose)

    def extract_phFCD(self, x):

        nt = x.shape[1]
        m = calc_FCD_ph(x)

        return build_matrix(m, nt-2)

    def phFCD(self, x, opt=None):
        '''
        calculate Phase FCD features

        Parameters
        ----------
        x: np.ndarray (2d)
            input signal array (n_channels, n_timepoints)
        opt: dict
            dictionary of parameters
            PCA_n_components: int
                number of PCA components to be used
            k: int
                number of subdiagonal to be excluded from FCD matrix

        Returns
        -------
        np.ndarray (1d)
            list of feature values
        '''
        if opt is None:
            opt = self.opt_phFCD

        k = opt['k']
        PCA_n_components = opt['PCA_n_components']
        nt = x.shape[1]
        ph = calc_FCD_ph(x)
        m = build_matrix(ph, nt-2)
        m = set_k_diogonal(m, k=k)
        FCD_TRIU = np.triu(m, k=k)
        off_diag_sum = np.sum(np.abs(m))

        eigvals, _ = LA.eig(m)
        pca = PCA(n_components=PCA_n_components)
        pca_v = pca.fit_transform(m)
        Lower_FCD = []
        for i in range(0, len(m)):
            Lower_FCD.extend(m[i][0:i])

        stats_vec = np.zeros(35)
        data = [eigvals.reshape(-1),
                pca_v.reshape(-1),
                Lower_FCD,
                FCD_TRIU.reshape(-1)]
        funcs = [np.sum, np.max, np.min, np.mean,
                 np.std, skew, kurtosis] * len(data)
        n0 = 7
        for ki in range(len(data)):
            for i in range(ki*n0, (ki+1)*n0):
                stats_vec[i] = funcs[i](data[ki])

        stats_vec = np.append(stats_vec, np.quantile(
            m, [0.05, 0.25, 0.5, 0.75, 0.95]))
        stats_vec = np.append(stats_vec, [off_diag_sum])

        return stats_vec

    def fcd_sum(self, FCD_res, opt=None):
        '''!
        Calculate the sum of functional connectivity dynamics (FCD)

        Parameters
        ----------

        FCD: np.ndarray (2d)
            input FCD matrix
        k: int
            number of subdiagonal to be excluded from FCD matrix
        Returns
        -------
        x: np.ndarray (1d)
            sum of functional connectivity dynamics
        '''

        if opt is None:
            opt = self.opt_fcd_sum

        k = opt['k']

        if len(FCD_res) == 1:  # find nan value
            return FCD_res
        else:
            FCDcorr, Pcorr, shift = FCD_res
            off_diag_sum_FCD = np.sum(
                np.abs(FCDcorr)) - np.trace(np.abs(FCDcorr))
            off_diag_sum_FCD = np.sum(set_k_diogonal(FCDcorr, k, 0.0))

            return off_diag_sum_FCD
    # -------------------------------------------------------------------------

    def fcd_corr(self, FCD_res, opt=None):
        '''!
        Calculate the freatures from functional connectivity dynamics (FCD)

        \param FCD np.ndarray (2d)
            input FCD matrix
        \param k_diagonal int
            number of subdiagonal to be excluded from FCD matrix
        \returns np.ndarray (1d)
            list of feature values
        '''

        if opt is None:
            opt = self.opt_fcd_corr

        k = opt['k']
        PCA_n_components = opt['PCA_n_components']
        demean = opt['demean']

        if len(FCD_res) == 1:  # find nan value
            return FCD_res
        else:
            FCDcorr, Pcorr, shift = FCD_res
            off_diag_sum_FCD = np.sum(
                np.abs(FCDcorr)) - np.trace(np.abs(FCDcorr))
            off_diag_sum_FCD = np.sum(set_k_diogonal(FCDcorr, k, 0.0))

            FCD_TRIU = np.triu(FCDcorr, k=k)  # ! bug fix (was k=10)

            eigen_vals_FCD, _ = LA.eig(FCDcorr)
            pca = PCA(n_components=PCA_n_components)
            PCA_FCD = pca.fit_transform(FCDcorr)

            Upper_FCD = []
            Lower_FCD = []
            for i in range(0, len(FCDcorr)):
                Upper_FCD.extend(FCDcorr[i][i+1:])
                Lower_FCD.extend(FCDcorr[i][0:i])

            # stats_vec = np.zeros(35)
            stats_vec = np.array([])
            if not demean:
                funcs = [np.sum, np.max, np.min, np.mean,
                         np.std, skew, kurtosis]
            else:
                funcs = [np.std, skew, kurtosis]
            data = [eigen_vals_FCD.reshape(-1),
                    PCA_FCD.reshape(-1),
                    Upper_FCD,
                    Lower_FCD,
                    FCD_TRIU.reshape(-1)]

            for ki in range(len(data)):
                _st = np.zeros(len(funcs))
                for i in range(len(funcs)):
                    _st[i] = funcs[i](data[ki])
                stats_vec = np.append(stats_vec, _st)

            # n0 = 7
            # for ki in range(len(data)):
            #     for i in range(ki*n0, (ki+1)*n0):
            #         print(funcs[i](data[ki]))
            #         stats_vec[i] = funcs[i](data[ki])

            stats_vec = np.append(stats_vec, np.quantile(
                FCDcorr, [0.05, 0.25, 0.5, 0.75, 0.95]))
            stats_vec = np.append(stats_vec, [off_diag_sum_FCD])

            return stats_vec

    # -------------------------------------------------------------------------
    def fcd_corr_regions(self, x, opt=None):
        '''!
        Calculate the freatures from functional connectivity dynamics (FCD) on given regions

        Parameters
        ----------
        x: np.ndarray (2d)
            input BOLD signal [nnodes, ntime]
        opt: dict
            options for fcd_corr_regions
            regions:  list[int]
                index of regions to use for FCD
            wwidth: int
                window width
            maxNwindows: int
                maximum number of windows
            olap: int
                overlap between windows
            mode: str
                mode of FCD
            k: int
                number of subdiagonal to be excluded from FCD matrix
            verbose: bool
                verbose mode
        Returns
        -------
        stats_vec: np.ndarray (1d)
            list of feature values
        '''

        if opt is None:
            opt = self.opt_fcd_corr_regions

        k = opt['k']
        olap = opt['olap']
        mode = opt['mode']
        wwidth = opt['wwidth']
        verbose = opt['verbose']
        regions = opt['regions']
        maxNwindows = opt['maxNwindows']

        assert(len(regions) >=
               2), 'regions should be at least 2, use set_feature_properties to set regions'
        assert(isinstance(regions[0], (np.int64, int, np.int32)))

        def funcs(x):
            vec = np.zeros(7)
            vec[0] = np.sum(x)
            vec[1] = np.max(x)
            vec[2] = np.min(x)
            vec[3] = np.mean(x)
            vec[4] = np.std(x)
            vec[5] = skew(x)
            vec[6] = kurtosis(x)
            return vec

        x_ = np.asarray(x)
        x_ = x_[regions, :]
        result = self.extract_FCD(x, wwidth, maxNwindows, olap,
                                  mode=mode, verbose=verbose)
        if len(result) == 1:  # find nan value
            return result
        else:
            FCDcorr, Pcorr, shift = result
            off_diag_sum_FCD = np.sum(
                np.abs(FCDcorr)) - np.trace(np.abs(FCDcorr))
            # !TODO check this line, probably wrong (how many diagonals to exclude?)
            off_diag_sum_FCD = np.sum(set_k_diogonal(FCDcorr, k, 0.0))
            FCD_TRIU = np.triu(FCDcorr, k=k)  # ! bug fix (was k=10)
            eigen_vals_FCD, _ = LA.eig(FCDcorr)
            pca = PCA(n_components=3)
            PCA_FCD = pca.fit_transform(FCDcorr)

            Upper_FCD = []
            Lower_FCD = []
            for i in range(0, len(FCDcorr)):
                Upper_FCD.extend(FCDcorr[i][i+1:])
                Lower_FCD.extend(FCDcorr[i][0:i])

            stats_vec = np.zeros(35)
            funcs = [np.sum, np.max, np.min, np.mean, np.std, skew, kurtosis]*5
            data = [eigen_vals_FCD.reshape(-1),
                    PCA_FCD.reshape(-1),
                    Upper_FCD,
                    Lower_FCD,
                    FCD_TRIU.reshape(-1)]
            n0 = 7
            for ki in range(len(data)):
                for i in range(ki*n0, (ki+1)*n0):
                    stats_vec[i] = funcs[i](data[ki])

            stats_vec = np.append(stats_vec, np.quantile(
                FCDcorr, [0.25, 0.5, 0.75]))
            stats_vec = np.append(stats_vec, [off_diag_sum_FCD])

            return stats_vec

    # -------------------------------------------------------------------------
    def fcd_elements(self, FCD, opt=None):
        '''!
        Calulate the functional connectivity dynamics (FCD)

        Parameters
        ----------
        FCD: np.ndarray (2d)
            input FCD matrix
        opt: dict
            parameters for FCD elements calculation
        Returns
        -------
        x: np.ndarray (1d)
            functional connectivity dynamics elements

        '''

        if opt is None:
            opt = self.opt_fcd_elements

        k = opt['k']

        if len(FCD) == 1:  # find nan value
            return FCD
        else:
            FCDcorr, Pcorr, shift = FCD
            FCDcorr = set_k_diogonal(FCDcorr, k, 0.0)
            return FCDcorr.reshape(-1)
    # -------------------------------------------------------------------------

    def extract_FCD(self, data, opt=None):
        """!
        Functional Connectivity Dynamics from a collection of time series

        Parameters
        ----------
        data: np.ndarray (2d)
            time series in rows [n_nodes, n_samples]
        opt: dict
            parameters

        Returns
        -------
        FCD: np.ndarray (2d)
            functional connectivity dynamics matrix

        """

        if opt is None:
            opt = self.opt_FCD_common

        mode = opt['mode']
        olap = opt['olap']
        axis = opt['axis']
        wwidth = opt['wwidth']
        verbose = opt['verbose']
        maxNwindows = opt['maxNwindows']

        assert(olap <= 1 and olap >= 0), 'olap must be between 0 and 1'

        if axis == 0:
            data = copy(data.T)

        all_corr_matrix = []
        lenseries = len(data[0])

        try:
            Nwindows = min(((lenseries-wwidth*olap) //
                            (wwidth*(1-olap)), maxNwindows))
            shift = int((lenseries-wwidth)//(Nwindows-1))
            if Nwindows == maxNwindows:
                wwidth = int(shift//(1-olap))

            indx_start = range(0, (lenseries-wwidth+1), shift)
            indx_stop = range(wwidth, (1+lenseries), shift)

            nnodes = len(data)

            for j1, j2 in zip(indx_start, indx_stop):
                aux_s = data[:, j1:j2]
                if mode == 'corr':
                    corr_mat = np.corrcoef(aux_s)
                elif mode == 'psync':
                    corr_mat = np.zeros((nnodes, nnodes))
                    for ii in range(nnodes):
                        for jj in range(ii):
                            corr_mat[ii, jj] = np.mean(
                                np.abs(np.mean(np.exp(1j*aux_s[[ii, jj], :]), 0)))
                elif mode == 'plock':
                    corr_mat = np.zeros((nnodes, nnodes))
                    for ii in range(nnodes):
                        for jj in range(ii):
                            corr_mat[ii, jj] = np.abs(
                                np.mean(np.exp(1j*np.diff(aux_s[[ii, jj], :], axis=0))))
                elif mode == 'tdcorr':
                    corr_mat = np.zeros((nnodes, nnodes))
                    for ii in range(nnodes):
                        for jj in range(ii):
                            maxCorr = np.max(np.correlate(aux_s[ii, :], aux_s[jj, :], mode='full')[
                                wwidth//2:wwidth+wwidth//2])
                            corr_mat[ii, jj] = maxCorr/np.sqrt(
                                np.dot(aux_s[ii, :], aux_s[ii, :])*np.dot(aux_s[jj, :], aux_s[jj, :]))
                all_corr_matrix.append(corr_mat)

            corr_vectors = np.array([allPm[np.tril_indices(nnodes, k=-1)]
                                    for allPm in all_corr_matrix])
            CV_centered = corr_vectors - np.mean(corr_vectors, -1)[:, None]

            return np.corrcoef(CV_centered), corr_vectors, shift

        except Exception as e:
            if verbose:
                print(e)
            return [np.nan]

    # -------------------------------------------------------------------------

    def fcd_regions_filt(self, bold, opt=None):
        '''
        extract features from FCD matrix of given region

        Parameters
        ----------
        bold: np.ndarray (2d)
            time series in rows [n_nodes, n_samples]
        opt: dict
            parameters

        Returns
        -------
        x: np.ndarray (1d)
            array of features value

        '''

        if opt is None:
            opt = self.opt_fcd_regions_filt

        k = opt['k']  # 1
        win_sp = opt['win_sp']
        verbose = opt['verbose']
        win_len = opt['win_len']
        regions_idx = opt['regions_idx']

        fcd_ = self.extract_fcd_regions_filt(bold,
                                             regions_idx,
                                             win_len=win_len,
                                             win_sp=win_sp,
                                             verbose=verbose)
        if len(fcd_) != 1:
            return([np.mean(np.triu(fcd_, k=k)), np.var(np.triu(fcd_, k=k))])
        else:
            return fcd_

    # -------------------------------------------------------------------------
    @staticmethod
    def compute_fcd_regions_filt(ts, mat_filt, win_len=30, win_sp=1, verbose=False):
        """Compute dynamic functional connectivity with SC filtering

        Arguments:
            ts:      time series of shape [time,nodes]
            win_len: sliding window length in samples
            win_sp:  sliding window step in samples

        Returns:
            FCD: matrix of functional connectivity dynamics
            fcs: windowed functional connectivity matrices
            speed_fcd: rate of changes between FC frames
        """
        n_samples, n_nodes = ts.shape
        # returns the indices for upper triangle
        fc_triu_ids = np.triu_indices(n_nodes, 1)
        n_fcd = len(fc_triu_ids[0])
        fc_stack = []
        speed_stack = []

        try:

            for t0 in range(0, ts.shape[0]-win_len, win_sp):
                t1 = t0+win_len
                fc = np.corrcoef(ts[t0:t1, :].T)
                fc = fc*(fc > 0)*(mat_filt)
                fc = fc[fc_triu_ids]
                fc_stack.append(fc)
                if t0 > 0:
                    corr_fcd = np.corrcoef([fc, fc_prev])[0, 1]
                    speed_fcd = 1-corr_fcd
                    speed_stack.append(speed_fcd)
                    fc_prev = fc
                else:
                    fc_prev = fc

            fcs = np.array(fc_stack)
            speed_ts = np.array(speed_stack)
            FCD = np.corrcoef(fcs)
            return FCD, fcs, speed_ts
        except Exception as e:
            if verbose:
                print(e)
            return [np.nan]

    def extract_fcd_regions_filt(self, bold, regions_idx, win_len, win_sp, verbose=False):
        '''
        extract fcd of given regions from bold signal

        \param bold 2d array
            bold signal [nnodes, ntime]
        \param regions_idx list of int
            indices of regions
        \param win_len int
            sliding window length in samples
        \param win_sp int
            sliding window step in samples
        \param k int
            number of diagonal to set to zero
        \param verbose bool
            verbose flag
        '''

        nn = bold.shape[0]
        maskregions = np.zeros((nn, nn))
        maskregions[np.ix_(regions_idx, regions_idx)] = 1  # making a mask

        result = self.compute_fcd_regions_filt(bold.T,
                                               maskregions,
                                               win_len=win_len,
                                               win_sp=win_sp,
                                               verbose=0)

        # check if return nan
        if len(result) == 1:
            if verbose:
                print(np.isnan(result).any())
            return result
        else:
            return result[0]

    def PSD_under_power(self, f, Pxx, opt=None):
        '''!
        Calculate the under-power of the power spectrum
        '''

        if opt is None:
            opt = self.opt_PSD_under_power

        axis = opt['axis']
        y_pwr = np.sum((Pxx*Pxx), axis=axis).reshape(-1)
        return (y_pwr / y_pwr.max()).reshape(-1)
    # -------------------------------------------------------------------------

    # def PSD_under_area(self, f, Pxx_den, opt=None):

    #     bands = {'delta': [0.5, 4],
    #              'theta': [4.0, 8.0],
    #              'alpha': [8.0, 15.0],
    #              'beta': [15.0, 30.0],
    #              'gamma': [30.0, 50.0],
    #              'high_gamma': [50.0, 150.0]}

    #     low_band = []
    #     high_band = []
    #     if opt is None:
    #         opt = self.opt_PSD_under_area

    #     assert(len(opt['bands']) > 0), "bands should not be empty."
    #     assert(all(elem in bands.keys()
    #            for elem in opt['bands']), "bands should be in the list of bands.")

    #     for band in opt['bands']:
    #         low_band.append(bands[band][0])
    #         high_band.append(bands[band][1])

    #     stat_vec = np.array([])
    #     for lo, hi in zip(low_band, high_band):
    #         idx = np.logical_and(f >= lo, f <= hi).tolist()
    #         if len(idx) > 0:
    #             area = np.trapz(Pxx_den[:, idx], f[idx], axis=1).reshape(-1)
    #             stat_vec = np.append(stat_vec, area)

    #     return stat_vec

    def PSD_under_area(self, f, pxx, opt=None):

        if opt is None:
            opt = self.opt_PSD_under_area

        avg = opt['average_over_channels']
        normalize = opt['normalize']

        fmin = opt['fmin']
        fmax = opt['fmax']

        if normalize:
            pxx = pxx/pxx.max()

        stat_vec = np.array([])
        idx = np.logical_and(f >= fmin, f <= fmax).tolist()
        if len(idx) > 0:
            if avg:
                pxx = np.mean(pxx, axis=0)
                area = np.trapz(pxx[idx], f[idx])
                stat_vec = np.append(stat_vec, area)
                return stat_vec

            else:
                area = np.trapz(pxx[:, idx], f[idx], axis=1).reshape(-1)
                stat_vec = np.append(stat_vec, area)
                return stat_vec
        else:
            return [np.nan]

    # -------------------------------------------------------------------------

    def welch(self, x, opt=None):
        '''!
        Estimate power spectral density using Welch's method.
        #!TODO check for 1d input signal
        Parameters
        ----------
        x: array-like [n_channels, n_samples]
            Time series
        fs: float
            sampling frequency of the data in Hz

        Returns
        -------
        f: np.ndarray
            frequency vector
        pxx: np.ndarray
            power spectral density vector

        '''
        if opt is None:
            opt = self.opt_PSD_common

        fs = opt['fs']
        nperseg = opt['nperseg']
        noverlap = opt['noverlap']

        freq, Pxx_den = signal.welch(
            x, fs=fs, nperseg=nperseg, noverlap=noverlap)

        return freq, Pxx_den
    # -------------------------------------------------------------------------

    def max_PSD_freq(self, freq, pxx, opt=None):
        '''!
        calculate the peak frequency of Power Specteral Density
        #!TODO check for 1d input signal

        Parameters
        ----------

        x: np.ndarray (2d)
            input signal [n_channels, n_samples]
        opt: dict
            parameters
        Returns
        -------
        x: np.ndarray (1d)
            peak frequency of the PSD
        '''

        if opt is None:
            opt = self.opt_max_PSD_freq

        avg = opt['average_over_channels']
        normalize = opt['normalize']

        if normalize:
            pxx = pxx/pxx.max()

        indices = np.argmax(pxx, axis=1).tolist()

        if avg:
            # return [freq[indices].mean()]
            pxx = np.mean(pxx, axis=0)
            idx = np.argmax(pxx)
            return [freq[idx], pxx[idx]]
        else:
            # output need to be list or array #!TODO check for 1d input signal
            return freq[indices], pxx[indices]
    # -------------------------------------------------------------------------

    def moments(self, x, opt=None):
        '''!
        calculate the moments of time series

        Parameters
        ----------
        x: np.ndarray (2d)
            input signal [n_channels, n_samples] with axis=1
        opt: dict
            parameters

        Returns
        -------
        x: np.ndarray (1d)
            moments of the time series
        '''

        if opt is None:
            opt = self.opt_moments
        axis = opt['axis']
        regions = opt['regions']
        demean = opt['demean']

        assert len(regions) <= x.shape[0], "len(regions) <= n nodes"

        if len(regions) > 0:
            x = x[regions, :]
            if len(x.shape) == 1:
                x = x.reshape(1, -1)

        m, n = x.shape
        nn = m if (axis == 1) else n

        if not demean:
            funcs = [np.mean, np.median, np.std, skew, kurtosis]
        else:
            funcs = [np.std, skew, kurtosis]

        nf = len(funcs)
        stats_vec = np.zeros(nf * nn)
        for i in range(len(funcs)):
            stats_vec[(i*nn):((i+1)*nn)] = funcs[i](x, axis=axis)

        return stats_vec
    # -------------------------------------------------------------------------

    def high_low_mu(self, y, opt=None):
        '''!
        Caculate the high-low-mean of a time series
        #!TODO check for axis=0

        Parameters
        ----------
        y: np.ndarray (2d)
            input signal [n_channels, n_samples] with axis=1
        opt: dict
            parameters

        Returns
        -------
        x: np.ndarray (1d)
            array of features
        '''

        if opt is None:
            opt = self.opt_high_low_mu

        axis = opt['axis']

        n = y.shape[0] if (axis == 1) else y.shape[1]

        result = np.zeros(n)

        for i in range(n):
            x = y[i, :]
            mu = np.mean(x)
            mhi = np.mean(x[np.where(x < mu)])
            mu = np.mean(x)
            mhi = np.mean(x[x > mu])
            mlo = np.mean(x[x < mu])
            result[i] = (mhi - mu) / (mu - mlo)

        return result
    # -------------------------------------------------------------------------

    def IQR(self, y, opt=None):
        '''!
        Compute the interquartile range of the data along the specified axis.

        Parameters
        ----------
        y: array-like
            2-D array of data
        opt: dict
            parameters
            axis: int
                axis along which to compute the interquartile range
        '''
        if opt is None:
            opt = self.opt_IQR
        axis = opt['axis']

        return iqr(y, axis=axis)
    # -------------------------------------------------------------------------

    def abs_moments(y, opt=None):
        '''!
        Compute the absolute moments of the data.
        '''

        sum_abs = np.sum(abs(y))
        mean_abs = np.mean(abs(y))
        median_abs = np.median(abs(y))
        peak_to_peak = abs(np.max(y)-np.min(y))

        return np.hstack([sum_abs, mean_abs, median_abs, peak_to_peak])
    # -------------------------------------------------------------------------

    def higher_moments(self, x, opt=None):
        '''!
        Compute the moments of the data.

        Parameters
        ----------
        x: array-like
            2-D array of data [n_channels, n_samples]
        opt: dict
            parameters
            axis: int
                axis along which to compute the higher moments

        Returns
        -------
        x: np.ndarray (1d)
            array of moments
        '''

        if opt is None:
            opt = self.opt_higher_moments
        
        axis = opt['axis']
        regions = opt['regions']

        assert len(regions) <= x.shape[0], "selected regions should be equal or less than the number of channels"
        if len(regions) > 0:
            x = x[regions, :]
            if len(x.shape) == 1:
                x = x.reshape(1, -1)

        m, n = x.shape
        nn = m if (axis == 1) else n

        stats_vec = np.zeros(9*nn)
        for i in range(9):
            stats_vec[(i*nn):((i+1)*nn)] = moment(x, moment=i+2, axis=axis)

        return stats_vec
    # -------------------------------------------------------------------------

    def extract_PSD(self, x, opt=None):
        '''!
        Extract the power spectral density of the input signal

        Parameters
        ----------
        x: np.ndarray (2d)
            input signal [n_channels, n_samples]
        opt: dict
            parameters

        Returns
        -------
        freq: np.ndarray (1d)
            frequency vector
        pxx: np.ndarray (2d)
            power spectral density vector [n_channels, n_freq_samples]

        '''

        if opt is None:
            opt = self.opt_PSD_common

        fs = opt['fs']
        method = opt['method']
        nperseg = opt['nperseg']
        noverlap = opt['noverlap']

        assert (not fs is None), 'frequency sampling (fs) is not specified.'
        if method == 'welch':
            freq, Pxx_den = signal.welch(x, fs=fs,
                                         nperseg=nperseg,
                                         noverlap=noverlap)
        elif method == 'periodogram':
            freq, Pxx_den = signal.periodogram(x, fs)

        return freq, Pxx_den
    # -------------------------------------------------------------------------

    def spectral_power(self, freq, pxx, opt=None):
        '''!
        Calculate features from the spectral power of the given signal.

        Parameters
        ----------
        x: array-like
            2-D array of data [n_channels, n_samples]
        opt: dict
            parameters
            fs: float
                sampling frequency of the data in Hz
            nperseg: int
                length of each segment
            axis: int
                axis along which to compute the spectral power

        Returns
        -------
        x: np.ndarray (1d)
            array of spectral power features
        '''

        if opt is None:
            opt = self.opt_spectral_power

        fs = opt['fs']
        avg = opt['average_over_channels']
        regions = opt['regions']

        if len(regions) > 0:
            pxx = pxx[regions, :]
            if len(pxx.shape) == 1:
                pxx = pxx.reshape(1, -1)

        if avg:
            pxx = np.mean(pxx, axis=0).reshape(1, -1)

        assert (not fs is None), 'frequency sampling (fs) is not specified.'

        stats_vec = []
        funcs = [np.max, np.mean, np.median, np.std, skew, kurtosis]
        for i in range(len(funcs)):
            stats_vec.append(funcs[i](pxx, axis=1).reshape(-1))

        return np.hstack(stats_vec)
    # -------------------------------------------------------------------------

    def envelope(self, x, opt=None):
        '''!
        Calculate the features from envelope of a signal using hilbert transform.

        \param x 2d_array_like
            The signal to be analyzed.
        \param axis
            The axis along which to calculate the envelope.

        \returns 1d_array_like
            list of features.
        '''

        if opt is None:
            opt = self.opt_envelope

        axis = opt['axis']
        regions = opt['regions']

        assert len(regions) <= x.shape[0], 'regions should be a list of channel indices.'
        if len(regions) > 0:
            x = x[regions, :]
            if len(x.shape) == 1:
                x = x.reshape(1, -1)

        m, n = x.shape
        nn = m if (axis == 1) else n
        # stats_vec = np.zeros(4*nn)
        stats_vec = np.array([])

        analytic_signal = hilbert(x)
        amplitude_envelope = np.abs(analytic_signal)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))

        funcs = [np.mean, np.std, np.mean, np.std]

        for i in range(2):
            stats_vec = np.append(stats_vec, funcs[i](
                amplitude_envelope, axis=axis))
            # stats_vec[(i*nn):((i+1)*nn)] = funcs[i](amplitude_envelope, axis=axis)
        for i in range(2, 4):
            stats_vec = np.append(stats_vec, funcs[i](
                instantaneous_phase, axis=axis))
            # stats_vec[(i*nn):((i+1)*nn)] = funcs[i](instantaneous_phase, axis=axis)

        return stats_vec
    # -------------------------------------------------------------------------

    def fcd_regions(self, x, opt=None):
        '''!
        Calculate the freatures from functional connectivity dynamics (FCD) on given regions
        #!TODO check for axis=0

        Parameters
        ----------
        x: array-like
            2-D array of data [n_channels, n_samples] with axis=1
        opt: dict
            parameters
            regions:  list[int]
                index of regions to use for FCD
            wwidth: int
                window width
            maxNwindows: int
                maximum number of windows
            olap: int
                overlap between windows
            mode: str
                mode of FCD, options: 'corr', 'psync', 'plock', 'tdcorr
            k: int
                number of subdiagonal to be excluded from FCD matrix
            verbose: bool
                verbose mode
        Returns
        -------
        x: np.ndarray (1d)
            list of feature values
        '''
        # regions, x0, wwidth, maxNwindows, olap, mode, k_diagonal=20, verbose=False):

        if opt is None:
            opt = self.opt_fcd_regions

        regions = opt['regions']

        assert(len(regions) >=
               2), 'regions should be at least 2, use set_feature_properties to set regions'
        assert(isinstance(regions[0], (np.int64, int, np.int32)))
        x = copy(x[regions, :])
        result = self.extract_FCD(x, opt)
        if len(result) == 1:  # find nan value
            return result
        else:
            FCDcorr, Pcorr, shift = result
            stats_vec = np.array([])
            stats_vec = np.append(stats_vec, np.mean(
                select_upper_triangular(FCDcorr, opt['k'])))
            stats_vec = np.append(stats_vec, np.var(
                select_upper_triangular(FCDcorr, opt['k'])))
        return stats_vec
    # -------------------------------------------------------------------------

    def raw_ts(self, x, opt=None):
        '''!
        Return flatten raw time series as feature.
        '''

        if opt is None:
            opt = self.opt_raw_ts

        x_ = np.asarray(x)
        return x_.flatten()

    def fcd_edge_var(self, bold, opt=None):
        if opt is None:
            opt = self.opt_fcd_edge_var

        result = compute_fcd_edge(bold)
        if len(result) == 1:  # find nan value
            return result
        else:
            return [np.var(result)]

    def fcd_edge(self, bold, opt=None):
        """!
        Compute the FCD edge from the BOLD time series.
        it is a wrapper of compute_fcd_edge()

        Parameters
        ----------
        bold: array-like
            BOLD time series of shape (n_nodes, n_timepoints)
        k: int
            Number of subdiganal to be excluded from the FCD matrix.
        Returns
        -------
        fcd_edge: array-like
            list of statistics of FCD edge matrix.
        """

        if opt is None:
            opt = self.opt_fcd_edge

        k = opt['k']
        PCA_n_components = opt['PCA_n_components']

        result = compute_fcd_edge(bold)

        if len(result) == 1:  # find nan value
            return result
        else:
            FCDe = result
            off_diag_sum_FCD = np.sum(np.abs(FCDe)) - np.trace(np.abs(FCDe))
            off_diag_sum_FCD = np.sum(set_k_diogonal(FCDe, k, 0.0))

            FCD_TRIU = np.triu(FCDe, k=k)

            eigen_vals_FCD, _ = LA.eig(FCDe)
            pca = PCA(n_components=PCA_n_components)
            PCA_FCD = pca.fit_transform(FCDe)

            Upper_FCD = []
            Lower_FCD = []
            for i in range(0, len(FCDe)):
                Upper_FCD.extend(FCDe[i][i+1:])
                Lower_FCD.extend(FCDe[i][0:i])

            stats_vec = np.zeros(35)
            funcs = [np.sum, np.max, np.min, np.mean, np.std, skew, kurtosis]*5
            data = [eigen_vals_FCD.reshape(-1),
                    PCA_FCD.reshape(-1),
                    Upper_FCD,
                    Lower_FCD,
                    FCD_TRIU.reshape(-1)]
            n0 = 7
            for ki in range(len(data)):
                for i in range(ki*n0, (ki+1)*n0):
                    stats_vec[i] = funcs[i](data[ki])

            stats_vec = np.append(stats_vec, np.quantile(FCDe,
                                                         [0.05, 0.25,
                                                          0.5, 0.75, 0.95]))
            stats_vec = np.append(stats_vec, [off_diag_sum_FCD])

            return stats_vec

    def coactivation(self, ts, opt={}):
        '''!
        Compute coactivation of the time series.
        '''

        _opts = {"moments": [2, 3, 4, 5],
                 "avg": False,    # if True return average over all/selected regions
                 "regions": [], }
        _opts.update(opt)

        avg = _opts["avg"]
        regions = _opts["regions"]
        moments = _opts["moments"]

        M = coactivation_degree(ts)

        stats_vec = np.array([])
        for i in moments:
            tmp = stats.moment(M, moment=i, axis=1)

            if len(regions) > 0:
                tmp = tmp[regions]

            if avg:
                tmp = np.mean(tmp)

            stats_vec = np.append(stats_vec, tmp)
        return stats_vec

    def coactivation_ph(self, ts, opt={}):
        '''!
        computes statistics of coactivation phase
        '''

        M = coactivation_phase(ts)
        funcs = [np.mean, np.median, np.std, skew, kurtosis]
        return np.array([f(M) for f in funcs])

    def local_to_global_coherence(self, x, opt=None):
        '''!
        Compute local-to-global coherence of the time series.
        '''

        if opt is None:
            opt = self.opt_local_to_global_coherence

        x_ = np.asarray(x)
        nn = x_.shape[0]
        edge_ts = go_edge(x_.T)
        f = local_to_global_coherence(edge_ts, nn, roi_indices=opt['regions'])
        return flatten(f)

    def KOP(self, x, opt=None):
        '''
        returns Kuramoto order parameter

        Parameters
        ----------
        x : array-like
            time series of shape (nodes, times).

        Returns
        -------
        float
            Kuramoto order parameter.
        '''
        if opt is None:
            opt = self.opt_KOP

        if opt['hilbert']:
            x_h = hilbert(x, axis=1)   
            x_phase = np.angle(x_h)
        else:
            x_phase = copy(x)
        r = np.abs(np.mean(np.exp(1j*x_phase), axis=0))

        return np.mean(r)

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def compute_cofluctuation(ts):
    """!
    Compute co-fluctuation (functional connectivity edge) time series for
    each pair of nodes by element-wise multiplication of z-scored node time
    series.


    @param ts : array_like
        Time series of shape (time, nodes).ts

    \return array_like
        Co-fluctuation (edge time series) of shape (time, node_pairs).
    """
    # T, N = ts.shape
    # cf = np.zeros((T, int(N * (N-1)/2)))
    # ts = stats.zscore(ts, axis=0)

    # for k, (i, j) in enumerate(zip(*np.triu_indices(ts.shape[1], 1))):
    #     cf[:, k] = ts[:, i] * ts[:, j]
    nt, nn = ts.shape
    ts = stats.zscore(ts, axis=0)

    pairs = np.triu_indices(nn, 1)
    cf = ts[:, pairs[0]] * ts[:, pairs[1]]

    return cf


def compute_fcd_edge(bold):
    """!
    Compute the FCD from the BOLD time series.

    \param bold : array_like
        BOLD time series of shape (nnodes, ntime)

    \return array_like
            matrix of edge functional connectivity dynamics of shape (time, time)
    """

    try:
        cf = compute_cofluctuation(bold.T)  # (time, node_pairs)
        return np.corrcoef(cf)              # (time, time)
    except:
        return [np.nan]


def coactivation_degree(ts, modality='new'):
    nn, nt = ts.shape
    ts = stats.zscore(ts, axis=1)
    if modality == 'correlation':
        global_signal = stats.zscore(np.mean(ts, axis=1))

    M = np.zeros((nn, nt))
    for i in range(nn):
        if modality != 'correlation':
            global_signal = np.mean(np.delete(ts, i, axis=0), axis=0)
        M[i] = ts[i, :]*global_signal
    return M


def coactivation_phase(ts):

    ts = stats.zscore(ts, axis=1)

    # phase global
    GS = np.mean(ts, axis=0)
    Phase = np.unwrap(np.angle(hilbert(GS)))
    Phase = (Phase + np.pi) % (2 * np.pi) - np.pi

    # phase regional
    phase_i = np.unwrap(np.angle(hilbert(ts, axis=1)), axis=1)
    phase_i = (phase_i + np.pi) % (2 * np.pi) - np.pi
    MSphase = np.mean(Phase - phase_i, axis=1)

    return MSphase



def fcd_mask(bold, regions_idx, win_len, win_sp, verbose=False):

    nn = bold.shape[0]
    maskregions = np.zeros((nn, nn))
    maskregions[np.ix_(regions_idx, regions_idx)] = 1  # making a mask

    result = compute_fcd_mask(bold.T,
                              maskregions,
                              win_len=win_len,
                              win_sp=win_sp,
                              verbose=0)

    if len(result) == 1:
        if verbose:
            print(np.isnan(result).any())
        return result
    else:
        return result[0]


def compute_fcd_mask(ts, mat_filt, win_len=30, win_sp=1, verbose=False):

    nt, nn = ts.shape
    fc_triu_ids = np.triu_indices(nn, 1)
    n_fcd = len(fc_triu_ids[0])
    fc_stack = []
    speed_stack = []

    try:

        for t0 in range(0, ts.shape[0]-win_len, win_sp):
            t1 = t0+win_len
            fc = np.corrcoef(ts[t0:t1, :].T)
            fc = fc*(fc > 0)*(mat_filt)
            fc = fc[fc_triu_ids]
            fc_stack.append(fc)
            if t0 > 0:
                corr_fcd = np.corrcoef([fc, fc_prev])[0, 1]
                speed_fcd = 1-corr_fcd
                speed_stack.append(speed_fcd)
                fc_prev = fc
            else:
                fc_prev = fc

        fcs = np.array(fc_stack)
        speed_ts = np.array(speed_stack)
        FCD = np.corrcoef(fcs)
        return FCD, fcs, speed_ts
    except Exception as e:
        if verbose:
            print(e)
        return [np.nan]


def fc_mask(sig, regions_idx, positive=False):

    nn, nt = sig.shape
    maskregions = np.zeros((nn, nn))
    maskregions[np.ix_(regions_idx, regions_idx)] = 1
    fc = np.corrcoef(sig)
    if positive:
        fc = fc *(fc > 0) * maskregions
    return fc * maskregions

def mat_stats(A, opt=None):
        '''!

                Parameters
        ----------
        x: np.ndarray (2d)
            input array
        opt: dict
            dictionary of parameters
        Returns
        -------
        stats: np.ndarray (1d)
            feature values
        '''
        demean = opt["demean"]

        def funcs(x, demean=False):
            if demean:
                vec = np.zeros(3)
                vec[0] = np.std(x)
                vec[1] = skew(x)
                vec[2] = kurtosis(x)

            else:
                vec = np.zeros(7)
                vec[0] = np.sum(x)
                vec[1] = np.max(x)
                vec[2] = np.min(x)
                vec[3] = np.mean(x)
                vec[4] = np.std(x)
                vec[5] = skew(x)
                vec[6] = kurtosis(x)
            return vec

        off_diag_sum_A = np.sum(np.abs(A)) - np.trace(np.abs(A))

        A_TRIU = np.triu(A, k=opt["k"])
        eigen_vals_A, _ = LA.eig(A)
        pca = PCA(n_components=opt["PCA_n_components"])
        PCA_A = pca.fit_transform(A)

        Upper_A = []
        Lower_A = []
        for i in range(0, len(A)):
            Upper_A.extend(A[i][i+1:])
            Lower_A.extend(A[i][0:i])

        q = np.quantile(A, [0.05, 0.25, 0.5, 0.75, 0.95])
        _stats = np.array([])
        _stats = np.append(_stats, q)
        _stats = np.append(_stats, funcs(Upper_A, demean))
        _stats = np.append(_stats, funcs(Lower_A, demean))
        _stats = np.append(_stats, funcs(PCA_A.reshape(-1), demean))
        _stats = np.append(_stats, funcs(A_TRIU.reshape(-1), demean))
        _stats = np.append(_stats, funcs(np.real(eigen_vals_A[:-1]), demean))
        _stats = np.append(_stats, [off_diag_sum_A])

        return _stats