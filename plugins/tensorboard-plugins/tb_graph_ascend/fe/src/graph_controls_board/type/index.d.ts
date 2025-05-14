
export interface MinimapVis {
    npu: boolean;
    bench: boolean;
}
export type Dataset = Array<RunItem>;

export type MetaDirType = {
    [key: string]: Array<string>
}


export interface UseMatchedType {
    saveMatchedNodesLink: (selection: any) => Promise<any>;
    addMatchedNodesLink: (
        npuNodeName: string,
        benchNodeName: string,
        selection: any,
    ) => Promise<MatchResultType>;
    deleteMatchedNodesLink: (
        npuNodeName: string,
        benchNodeName: string,
        selection: any,
    ) => Promise<MatchResultType>;
    saveMatchedRelations: (selection: any) => Promise<any>;
    addMatchedNodesLinkByConfigFile: (condfigFile: string, selection: any) => Promise<MatchResultType>;
}
export type MatchResultType = {
    success: boolean;
    error: string;
    data?: {
        'npuMatchNodes': { ['string']: string };
        'benchMatchNodes': { ['string']: string };
        'npuUnMatchNodes': Array<string>;
        'benchUnMatchNodes': Array<string>;
        'matchReslut'?: Array<Boolean>
    }
}
