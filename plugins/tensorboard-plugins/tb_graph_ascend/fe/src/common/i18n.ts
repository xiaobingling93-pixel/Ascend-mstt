// i18n.js
import i18next from 'i18next';
import LanguageDetector from 'i18next-browser-languagedetector';

i18next
    .use(LanguageDetector)
    .init({
        fallbackLng: 'zh-CN',
        resources: {
            en: {
                translation: {
                    "fit": "Fit Screen",
                    "settings": "Settings",
                    "match": 'Matching',
                    "show_debug_minimap": "show debug minimap",
                    "show_bench_minimap": "show bench minimap",
                    "run": "Run",
                    "tag": "Tag",
                    "invalid_rank_id": "Tip: The target file does not exist"
                }
            },
            'zh-CN': {
                translation: {
                    "fit": "自适应屏幕",
                    "settings": "设置",
                    "match": '匹配',
                    "show_debug_minimap": "调试侧缩略图",
                    "show_bench_minimap": "标杆侧缩略图",
                    "run": "目录",
                    "tag": "文件",
                    "invalid_rank_id": "提示：目标文件不存在"
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