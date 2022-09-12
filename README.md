# IDS-FL

You can get the dataset [from here](https://www.unb.ca/cic/datasets/ids-2017.html)
after making one `csv` file from all of the `csv` files you can feed it to the `prepare_dataset` function.

### Feature Selection
<details>
<summary>Features:</summary>

we select the features according to [this study](https://www.scitepress.org/Link.aspx?doi=10.5220/0006639801080116) for [this dataset](https://www.unb.ca/cic/datasets/ids-2017.html)

#### The selected features are:

- For DoS GoldenEye Attack:
  - Bwd Packet Length Std
  - Flow IAT Min
  - Fwd IAT Min
  - Flow IAT Mean
- For DoS Hulk Attack:
  - Bwd Packet Len Std
  - Flow Duration
  - Flow IAT Std
- For DoS Slowhttp Attack:
  - Flow Duration
  - Active Min
  - Active Mean
  - Flow IAT Std
- For DoS slowloris arrack:
  - Flow Duration
  - Flow IAT Min
  - Bwd IAT Mean
  - Flow IAT Mean
- For DDoS Attack:
  - Bwd Packet Len Std
  - Avg Packet Size
  - Flow Duration
  - Flow IAT Std
</details>
