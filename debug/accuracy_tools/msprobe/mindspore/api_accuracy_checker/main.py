from msprobe.mindspore.api_accuracy_checker.api_accuracy_checker import ApiAccuracyChecker

def add_api_accuracy_checker_argument(parser):
    parser.add_argument("-api_info", "--api_info_file", dest="api_info_file", type=str, required=True,
                        help="<Required> The api param tool result file: generate from api param tool, "
                             "a json file.")
    parser.add_argument("-o", "--out_path", dest="out_path", default="./", type=str, required=False,
                        help="<optional> The ut task result out path.")


def api_checker_main(args):
    api_accuracy_checker = ApiAccuracyChecker()
    api_accuracy_checker.parse(args.api_info_file)
    api_accuracy_checker.run_and_compare()
    api_accuracy_checker.to_detail_csv(args.out_path)
    api_accuracy_checker.to_result_csv(args.out_path)