import '@vaadin/checkbox';
import '@vaadin/confirm-dialog'
import '@vaadin/checkbox-group';
import '@vaadin/text-field';
import { customElement, observe, property } from '@polymer/decorators';
import { html, PolymerElement } from '@polymer/polymer';
import { LegacyElementMixin } from '../../../polymer/legacy_element_mixin';
import request from '../../../utils/request';

@customElement('tf-filter-precision-error')
class TfFilterPrecisionError extends LegacyElementMixin(PolymerElement) {
    static readonly template = html`
        <vaadin-confirm-dialog
          id="filter-dialog"
          header="筛选精度误差计算指标"
          cancel-button-visible
          cancel-text="取消"
          confirm-text="确认"
          confirm-theme=" primary"
          opened="{{filterDialogOpened}}"
          confirm="[[onFlterDialogConfirm]]"
        >
            <vaadin-checkbox-group
                label="计算指标"
                value="{{filterValue}}"
                theme="vertical"
            >
                <vaadin-checkbox value="[[MAX_RELATIVE_ERR]]" label="MaxRelativeErr"></vaadin-checkbox>
                <vaadin-checkbox value="[[MIN_RELATIVE_ERR]]" label="MinRelativeErr"></vaadin-checkbox>
                <vaadin-checkbox value="[[MEAN_RELATIVE_ERR]]" label="MeanRelativeErr"></vaadin-checkbox>
                <vaadin-checkbox value="[[NORM_RELATIVE_ERR]]" label="NormRelativeErr"></vaadin-checkbox>
            </vaadin-checkbox-group>
        </vaadin-confirm-dialog>
    `

    @property({ type: Boolean, notify: true })
    filterDialogOpened: boolean = false;

    @property({ type: Array })
    filterValue: string[] = [];

    MAX_RELATIVE_ERR = "0";
    MIN_RELATIVE_ERR = "1";
    MEAN_RELATIVE_ERR = "2";
    NORM_RELATIVE_ERR = "3";
    override ready(): void {
        super.ready();
        const filterDialog = this.shadowRoot?.querySelector('#filter-dialog') as HTMLElement;
        filterDialog?.addEventListener('confirm', this.onFlterDialogConfirm)
        this.set('filterValue', [this.MAX_RELATIVE_ERR, this.MIN_RELATIVE_ERR, this.MEAN_RELATIVE_ERR, this.NORM_RELATIVE_ERR]);
    }
    onFlterDialogConfirm = async (e: any) => {
        this.set('filterDialogOpened', false);
        const data = {
            filterValue: (this.filterValue)
        };
        const mactchResult = await request({ url: 'updatePrecisionError', method: 'POST', data });
    }

}