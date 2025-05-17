export interface ProgressType {
    progress?: number;
    progressValue?: number;
    size?: number;
    read?: number;
    done?: boolean;
}

export interface SelectionType {
    run: string;
    tag: string;
    microStep: number;

}

export interface GraphConfigType {
    tooltips: string;
    colors: { string: { value: number[], color: string } },
    overflowCheck: boolean;
    microSteps: number;
    isSingleGraph: boolean;
    matchedConfigFiles: string[];
}

export interface GraphAllNodeType {
    npuNodeList: string[];
    benchNodeList: string[];
    npuUnMatchNodes: string[];
    benchUnMatchNodes: string[];
    npuMatchNodes: string[];
    benchMatchNodes: string[];
}