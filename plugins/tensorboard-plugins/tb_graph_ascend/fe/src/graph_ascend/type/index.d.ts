export interface ProgressType {
    progress?: number;
    progressValue?: number;
    size?: boolean;
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
    colors: object;
    overflowCheck: boolean;
    microSteps: number;
    isSingleGraph: boolean;
    matchedConfigFiles: Array<string>;
}

export interface GraphAllNodeType {
    npuNodeList: string[];
    benchNodeList: string[];
    npuUnMatchNodes: string[];
    benchUnMatchNodes: string[];
    npuMatchNodes: string[];
    benchMatchNodes: string[];
}