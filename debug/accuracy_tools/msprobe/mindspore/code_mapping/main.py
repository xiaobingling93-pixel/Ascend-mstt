from msprobe.mindspore.code_mapping.processor import process
from msprobe.mindspore.code_mapping.cmd_parser import check_args


def code_mapping_main(args):
    check_args(args)
    process(args)


if __name__ == "__main__":
    code_mapping_main()
