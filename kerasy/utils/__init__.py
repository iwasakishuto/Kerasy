# coding: utf-8
from __future__ import absolute_import
import os

UTILS_DIR_PATH = os.path.dirname(os.path.abspath(__file__))

class KerasyImprementationWarning(Warning):
    """Implementation problem. Developer will resolve it."""
    pass

from . import bio_utils
from . import deep_utils
from . import generic_utils
from . import hash_utils
from . import metric_utils
from . import model_select_utils
from . import monitor_utils
from . import np_utils
from . import param_utils
from . import prepro_utils
from . import toydata_utils
from . import training_utils
from . import tree_utils
from . import vis_utils

# Globally-importable utils.
from .bio_utils import bpHandler
from .bio_utils import arangeFor_printAlignment
from .bio_utils import printAlignment
from .bio_utils import read_fastseq
from .bio_utils import kmer_create

from .deep_utils import set_weight
from .deep_utils import mk_class_get
from .deep_utils import get_params_size
from .deep_utils import print_summary

from .generic_utils import flatten_dual
from .generic_utils import get_varname
from .generic_utils import disp_var_globals
from .generic_utils import disp_val_globals
from .generic_utils import get_file
from .generic_utils import priColor
from .generic_utils import handleKeyError
from .generic_utils import handleTypeError
from .generic_utils import urlDecorate
from .generic_utils import measure_complexity
from .generic_utils import has_all_attrs
from .generic_utils import has_not_attrs
from .generic_utils import handleRandomState
from .generic_utils import fout_args
from .generic_utils import format_spec_create
from .generic_utils import print_func_create
from .generic_utils import Table

from .hash_utils import optimize_hash_func
from .hash_utils import optimize_bit_format
from .hash_utils import make_hashfuncs

from .metric_utils import norm_vectors
from .metric_utils import normalize_vectors
from .metric_utils import mean_squared_error
from .metric_utils import root_mean_squared_error
from .metric_utils import silhouette_samples
from .metric_utils import silhouette_score
from .metric_utils import check_paired_array
from .metric_utils import paired_distances
from .metric_utils import paired_euclidean_distances
from .metric_utils import paired_manhattan_distances
from .metric_utils import paired_cosine_distances
from .metric_utils import check_pairwise_array
from .metric_utils import pairwise_distances
from .metric_utils import pairwise_euclidean_distances

from .model_select_utils import cross_validation

from .monitor_utils import ConvergenceMonitor
from .monitor_utils import ProgressMonitor
from .monitor_utils import ThresholdMonitor

from .np_utils import CategoricalEncoder
from .np_utils import findLowerUpper
from .np_utils import inverse_arr
from .np_utils import _check_sample_weight
from .np_utils import normalize
from .np_utils import standardize
from .np_utils import log_normalize
from .np_utils import log_mask_zero
from .np_utils import log_multivariate_normal_density
from .np_utils import compress_based_on_covariance_type_from_tied_shape
from .np_utils import decompress_based_on_covariance_type

from .param_utils import Params

from .prepro_utils import basis_transformer

from .progress_utils import flush_progress_bar
from .progress_utils import flush_progress_plot

from .toydata_utils import generateX
from .toydata_utils import generateSin
from .toydata_utils import generateGausian
from .toydata_utils import generateMultivariateNormal
from .toydata_utils import generateWholeCakes
from .toydata_utils import generateWhirlpool
from .toydata_utils import generateSeq
from .toydata_utils import generateSeq_embedded_Motif
from .toydata_utils import generate_test_data

from .training_utils import make_batches
from .training_utils import train_test_split
from .training_utils import iter_from_variable_len_samples

from .tree_utils import DOTexporterHandler
from .tree_utils import DecisionTreeDOTexporter
from .tree_utils import ItemsetTreeDOTexporter
from .tree_utils import NaiveTrieDOTexporter

from .vis_utils import rgb2hex, rgba2rgb, hex2rgb
from .vis_utils import chooseTextColor
from .vis_utils import mk_color_dict
from .vis_utils import measureCanvas
from .vis_utils import galleryplot
from .vis_utils import plot_2D
from .vis_utils import objVSexp
from .vis_utils import silhouette_plot
from .vis_utils import _left_silhouette_plot
from .vis_utils import _right_silhouette_plot
from .vis_utils import plot_CDF
from .vis_utils import plot_roc_curve
