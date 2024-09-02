from msprobe.mindspore.api_accuracy_checker.api_accuracy_checker import ApiAccuracyChecker


def api_checker_main(args):
    api_accuracy_checker = ApiAccuracyChecker()
    api_accuracy_checker.parse(args.api_info_file)
    api_accuracy_checker.run_and_compare()
    api_accuracy_checker.to_detail_csv(args.out_path)
    api_accuracy_checker.to_result_csv(args.out_path)