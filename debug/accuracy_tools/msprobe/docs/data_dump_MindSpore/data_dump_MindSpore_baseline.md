# MindSpore 场景的精度数据采集基线

## "tensor"模式采集数据量参考基线

该基线为MindSpore框架下，使用"tensor"模式采集数据量参考基线。本基线测试了38B语言大模型在不同采集模式下，不同global_batch_size下，单卡和8卡下，数据量的变化。

### 38B语言大模型

<table>
    <tr><th>采集模式</th><th>global_batch_size</th><th>单卡</th><th>8卡</th></tr>
    </td><td rowspan="3">L0</td><td>1</td><td>262GB</td><td>2.1T</td></tr>
    <tr><td>2</td><td>480GB</td><td>3.8T</td></tr>
    <tr><td>3</td><td>928GB</td><td>7.4T</td></tr>
    </td><td rowspan="3">L1</td><td>1</td><td>2.1TB</td><td>17.1TB</td></tr>
    <tr><td>2</td><td>2.8T</td><td>22.7TB</td></tr>
    <tr><td>3</td><td>4.2T</td><td>34.3TB</td></tr>
    </td><td rowspan="3">mix</td><td>1</td><td>2.4T</td><td>19.2TB</td></tr>
    <tr><td>2</td><td>3.3TB</td><td>26.6TB</td></tr>
    <tr><td>3</td><td>5.1TB</td><td>41.4TB</td></tr>

</table>

