/* Copyright (c) 2025, Huawei Technologies.
 * All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0  (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
import i18next from 'i18next';
import LanguageDetector from 'i18next-browser-languagedetector';

i18next
    .use(LanguageDetector)
    .init({
        fallbackLng: 'zh-CN',
        resources: {
            'en': {
                translation: {
                    fit: "Fit Screen",
                    settings: "Settings",
                    match: 'Matching',
                    show_debug_minimap: "show debug minimap",
                    show_bench_minimap: "show bench minimap",
                    run: "Run",
                    tag: "Tag",
                    invalid_rank_id: "Tip: The target file does not exist",
                    data_side: "Data Side",
                    search_node: "Search Node",
                    node_list: "Node List",
                    debug: "Debug",
                    bench: "Bench",
                    accuracy_error: "Accuracy Error",
                    overflow: "Overflow",
                    match_accuracy_error: "Match Accuracy Error Node",
                    overflow_filter_node: "Overflow Filter Node",
                    no_matching_nodes: "No matching nodes",
                    precision_desc: {
                        summary: "The relative error between the statistical output of the debug side and the benchmark side of the node, the larger the value, the greater the precision gap, the darker the color mark, the relative error indicator (RelativeErr): | (debug value - benchmark value) / benchmark value |",
                        all: "The difference between the minimum double thousand indicator of all inputs and the minimum double thousandth indicator of all outputs of the node, reflecting the decline of the double thousand indicator, the larger the value, the greater the precision gap, the darker the color mark, the double thousandth precision indicator (One Thousandth Err Ratio): The relative error of each element in the tensor is compared with the corresponding benchmark data, the proportion of relative error less than one thousandth of the total number of elements, the closer the proportion is to 1, the better",
                        md5: "If the md5 value of any input or output of the node is different, it will be marked red"
                    },
                    node_match: "Node Match",
                    select_match_config_file: "Select Match Config File",
                    select_match_config_file_desc: "Select the corresponding configuration file, read the matching node information, and match the corresponding node."

                }
            },
            'zh-CN': {
                translation: {
                    fit: "自适应屏幕",
                    settings: "设置",
                    match: '匹配',
                    show_debug_minimap: "调试侧缩略图",
                    show_bench_minimap: "标杆侧缩略图",
                    run: "目录",
                    tag: "文件",
                    invalid_rank_id: "提示：目标文件不存在",
                    data_side: "数据侧",
                    search_node: "节点搜索",
                    node_list: "节点列表",
                    debug: "调试侧",
                    bench: "标杆侧",
                    accuracy_error: "精度误差",
                    overflow: "精度溢出",
                    match_accuracy_error: "符合精度误差节点",
                    overflow_filter_node: "溢出筛选节点",
                    no_matching_nodes: "无匹配节点11",
                    precision_desc: {
                        "summary": "节点中调试侧和标杆侧输出的统计量相对误差，值越大精度差距越大，颜色标记越深,相对误差指标（RelativeErr）：| (调试值 - 标杆值) / 标杆值 |",
                        "all": "节点中所有输入的最小双千指标和所有输出的最小双千分之一指标的差值，反映了双千指标的下降情况，值越大精度差距越大，颜色标记越深，双千分之一精度指标（One Thousandth Err Ratio）：Tensor中的元素逐个与对应的标杆数据对比，相对误差小于千分之一的比例占总元素个数的比例，比例越接近1越好",
                        "md5": "节点中任意输入输出的md5值不同则标记为红色"
                    },
                    node_match: "节点匹配",
                    select_match_config_file: "选择匹配配置文件",
                    select_match_config_file_desc: "选择对应配置文件，会读取匹配节点信息，并将对应节点进行匹配。"
                }
            }
        },
        detection: {
            order: ['navigator'] // 只使用浏览器语言检测
        },
        debug: false,
        interpolation: {
            escapeValue: false
        }
    });

export default i18next;