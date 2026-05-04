# Parallelism in Transformers: My Implementation
![parallelism](./images//README.png)

**Тема ВКР:** Оптимизация распределённого инференса больших языковых моделей с архитектурой Mixture of Experts

### На данный момент реализовано три типа параллелизма:
 1. Tensor Parallel
 2. Pipeline Parallel с microbatch (2.0) и возможностью использования с Tensor Parallel
 3. MoE Parallel с возможностью использования с Data Parallel

 ## Эксперименты

### Сравнение реализаций тензорного и конвейерного параллелизма с DeepSpeed

| Параметр                                 | DeepSpeed      | New partitionig      | Standard Partitioning      | Pipeline |
|------------------------------------------|----------------|----------------------|---------------------------|----------|
| **Время инференса (s)**                  | 592.807        | 1988.417             | 757.441                   | 651.239  |
| **Пропускная способность (token/s)**     | 4.750          | 1.416                | 3.718                     | 4.717    |
| **Макс. память на устройство (генератор, ГБ)** | device 0: 2.706<br>device 1: 2.706 | device 0: 2.651<br>device 1: 2.651 | device 0: 2.651<br>device 1: 2.651 | device 0: 2.650<br>device 1: 2.635 |
| **Макс. память на устройство (инференс, ГБ)** | device 0: 3.400<br>device 1: 3.400 | device 0: 3.333<br>device 1: 3.333 | device 0: 4.331<br>device 1: 4.331 | device 0: 3.407<br>device 1: 3.391 |
| **Макс. память на устройство (общая, ГБ)**   | device 0: 3.400<br>device 1: 3.400 | device 0: 3.333<br>device 1: 3.333 | device 0: 4.331<br>device 1: 4.331 | device 0: 3.407<br>device 1: 3.391 |

### Сравнение конвейерного параллелзма 1.0 и 2.0 при generate на opt-1.3b

![1.0-vs-2.0](./images/results/opt-1.3b.svg)
![1.0-vs-2.0](./images/results/opt-1.3b-prefill.svg)
![1.0-vs-2.0](./images/results/pareto_analysis.svg)