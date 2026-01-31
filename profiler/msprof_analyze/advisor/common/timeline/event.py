#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is part of the MindStudio project.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#    http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

from decimal import Decimal


class AdvisorDict(dict):
    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)

    def __getattr__(self, key: str):
        if key not in self:
            return {}

        value = self[key]
        if isinstance(value, dict):
            value = AdvisorDict(value)
        return value


class TimelineEvent(AdvisorDict):

    def ts_include(self, event):
        self_ts = self.ts
        event_ts = event.ts

        if not self_ts or not event_ts:
            return False

        self_dur = self.dur if not isinstance(self.dur, dict) else 0.0
        event_dur = event.dur if not isinstance(event.dur, dict) else 0.0

        return Decimal(self_ts) <= Decimal(event_ts) and Decimal(self_ts) + Decimal(self_dur) >= Decimal(
            event_ts) + Decimal(event_dur)
