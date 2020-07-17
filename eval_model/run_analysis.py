#!/usr/bin/env python
# -*- encoding:utf-8 -*-
from analysis import analysis
import logging

file_name = "/search/odin/liruihong/TextEncoder/data_sets/annotate_data/kw_title_online_socre"
ndcg_val = analysis.cal_dumpfile_NDCG(file_name, 10)
print("%s ndcg_val:%f"%(file_name, ndcg_val))
