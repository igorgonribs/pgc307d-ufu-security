# pgc307d-ufu-security
Trabalho de conclusão da disciplina Segurança da Informação e Criptografia do Programa de Pós Graduação da Universidade Federal de Uberlândia

## Sobre o trabalho
O objetivo deste projeto é treinar diferentes modelo de aprendizagem de máquina para classificar tráfego de rede como Benigno ou ataque DDoS.

O dataset utilizado para treinamento e validação está disponível em https://www.unb.ca/cic/datasets/ids-2017.html.

Os modelos foram treinados com 70% , e validados com 30% do dataset.

Os resultados dos modelos serão comparados avaliando o **tempo de treinamento** e **taxa de acerto** gerados para cada algoritmo.

Os algoritmos testados são **Random Forest, Gradient Boosting, K Neighbors e Decision Tree**.

## Getting Started
Primeiramente verifique a instalação das seguintes dependências:
```bash
python3
sklearn
pandas
```

Uma vez que essas dependencias estão presentes no ambiente, baixe este repositório e execute o seguinte comando:
```bash
python3 classification_script.py
```

Nesse caso, o treinamento será realizaod com todos os algoritmos suportados.

Após a execução, a aplicação ira gerar uma saída semelhante à seguinte:
![image](https://user-images.githubusercontent.com/41350310/133942192-1c2fd398-bd8d-4846-a488-136e6998f8c6.png)

Você também pode especificar qual algoritmo deseja utilizar informando-o na linha de comando.
Exemplo: 
```bash
python3 classification_script.py knn
```

Os algoritmos suportados podem ser vistos com o comando:
```bash
python3 classification_script.py help
Supported arguments:
rf    - Random Forest
boost - Gradient Boosting
nb    - Gaussian NB
knn   - K Neighbors
dt    - Decision Tree

```
