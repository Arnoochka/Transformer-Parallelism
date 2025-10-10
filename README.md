# Parallelism in Transformers: My Implementation
![loss](./images//README.png)

| Параметр                                 | DeepSpeed      | New partitionig      | Standard Partitioning      |
|------------------------------------------|--------------------|--------------------|--------------------|
| **Время инференса (s)**                  | 592.807            | 1988.417           | 757.441            |
| **Пропускная способность (token/s)**     | 4.750              | 1.416              | 3.718              |
| **Макс. память на устройство (генератор, ГБ)** | device 0: 2.706<br>device 1: 2.706 | device 0: 2.651<br>device 1: 2.651 | device 0: 2.651<br>device 1: 2.651 |
| **Макс. память на устройство (инференс, ГБ)** | device 0: 3.400<br>device 1: 3.400 | device 0: 3.333<br>device 1: 3.333 | device 0: 4.331<br>device 1: 4.331 |
| **Макс. память на устройство (общая, ГБ)**   | device 0: 3.400<br>device 1: 3.400 | device 0: 3.333<br>device 1: 3.333 | device 0: 4.331<br>device 1: 4.331 |



