# coding: utf-8
import sys
import warnings
from kerasy.utils import KerasyImprementationWarning

def pytest_addoption(parser):
    parser.addoption("--kerasy-warnings", choices=["error", "ignore", "always", "default", "module", "once"], default="ignore")

def pytest_configure(config):
    action = config.getoption("kerasy_warnings")
    warnings.simplefilter(action, category=KerasyImprementationWarning)
