# sassamaru-25.py

Previsor de resultados do Campeonato Brasileiro utilizando modelo híbrido Poisson + ELO, com ajuste por forma recente e ponderação de resultados baseada na diferença de ELO.

## Principais Funcionalidades

- Cálculo de forças de ataque/defesa dos times via Poisson.
- Atualização dinâmica dos ratings ELO dos times, considerando vantagem de casa, diferença de gols, posição na tabela e peso pela diferença de ELO.
- Ajuste das probabilidades de resultado conforme a forma recente dos times.
- Geração de previsões para partidas futuras, exibindo probabilidades, palpites, ELOs e forma recente.

## Dependências

- pandas
- numpy
- scipy
- matplotlib
- scikit-learn

Instale as dependências com:
```bash
pip install pandas numpy scipy matplotlib scikit-learn
```

## Estrutura do Código

### Parâmetros Globais

- `ELO_RATING_INICIAL`: Valor inicial do ELO para todos os times.
- `ELO_K_FACTOR_BASE`: Fator base de ajuste do ELO.
- `ELO_VANTAGEM_CASA_PADRAO`: Vantagem padrão para o time mandante.
- `POISSON_MAX_GOLS`: Máximo de gols considerado na distribuição de Poisson.
- `ELO_INFLUENCE`: Influência do ELO no ajuste de gols esperados.
- `RECENT_FORM_GAMES`: Número de jogos recentes considerados para forma.
- `FORM_ADJUSTMENT_FACTOR`: Quanto a forma recente influencia as probabilidades.

---

## Funções Principais

### `calcular_forcas_poisson(df)`
Calcula as forças de ataque e defesa (casa/fora) de cada time usando médias de gols.

- **Parâmetros:**  
  `df` (pd.DataFrame): DataFrame com colunas 'mandante', 'visitante', 'gols_mandante', 'gols_visitante'
- **Retorna:**  
  `forcas` (dict): Forças de ataque/defesa para cada time  
  `medias_liga` (dict): Médias de gols da liga (casa/fora)

---

### `calcular_vantagens_casa(df)`
Calcula a vantagem de jogar em casa para cada time, baseada no saldo médio de gols como mandante.

- **Parâmetros:**  
  `df` (pd.DataFrame): DataFrame de partidas
- **Retorna:**  
  `vantagens` (dict): Vantagem de casa para cada time

---

### `calcular_forma_recente(df, n_jogos=5)`
Calcula a forma recente (vitórias, empates, derrotas) dos times nos últimos n_jogos.

- **Parâmetros:**  
  `df` (pd.DataFrame): DataFrame de partidas  
  `n_jogos` (int): Número de jogos recentes a considerar
- **Retorna:**  
  `forma_recente` (dict): Dicionário com contagem de resultados recentes para cada time

---

### `poisson(lmbda, k)`
Calcula a probabilidade de marcar k gols dado lambda (média esperada) usando a distribuição de Poisson.

- **Parâmetros:**  
  `lmbda` (float): Média esperada de gols  
  `k` (int): Número de gols
- **Retorna:**  
  `float`: Probabilidade de marcar k gols

---

### `prever_partida_hibrido(time_casa, time_visitante, context)`
Calcula as probabilidades de vitória, empate e derrota para uma partida usando modelo híbrido Poisson + ELO.

- **Parâmetros:**  
  `time_casa` (str): Nome do time mandante  
  `time_visitante` (str): Nome do time visitante  
  `context` (dict): Contexto com ratings ELO, forças Poisson, médias da liga e vantagens de casa
- **Retorna:**  
  `dict`: Probabilidades, gols esperados, ELOs e palpite para a partida

---

### `prever_partida_hibrido_com_forma(time_casa, time_visitante, context, forma_recente)`
Ajusta as probabilidades do modelo híbrido considerando a forma recente dos times.

- **Parâmetros:**  
  `time_casa` (str): Nome do time mandante  
  `time_visitante` (str): Nome do time visitante  
  `context` (dict): Contexto do modelo  
  `forma_recente` (dict): Forma recente dos times
- **Retorna:**  
  `dict`: Probabilidades ajustadas, ELOs, gols esperados, forma recente e palpite

---

### `calcular_peso_elo(rating_vencedor, rating_perdedor)`
Calcula o peso de uma vitória baseada na diferença de ELO. Vitórias improváveis (ELO menor vence ELO maior) têm peso maior.

- **Parâmetros:**  
  `rating_vencedor` (float): ELO do time vencedor  
  `rating_perdedor` (float): ELO do time perdedor
- **Retorna:**  
  `float`: Fator de peso para multiplicar o K

---

### `atualizar_ratings_elo(rating_c, rating_v, placar_c, placar_v, vantagem_c, df_standings, time_c, time_v)`
Atualiza os ratings ELO dos times após uma partida, considerando vantagem de casa, diferença de gols, posição na tabela e peso pela diferença de ELO.

- **Parâmetros:**  
  `rating_c` (float): ELO do mandante antes do jogo  
  `rating_v` (float): ELO do visitante antes do jogo  
  `placar_c` (int): Gols do mandante  
  `placar_v` (int): Gols do visitante  
  `vantagem_c` (float): Vantagem de casa do mandante  
  `df_standings` (pd.DataFrame): DataFrame com posições dos times  
  `time_c` (str): Nome do mandante  
  `time_v` (str): Nome do visitante
- **Retorna:**  
  `tuple`: (novo_rating_c, novo_rating_v) - Novos ratings ELO para mandante e visitante

---

## Fluxo Principal (`main`)

- Carrega dados do campeonato e classificação.
- Calcula forças, vantagens, forma recente e inicializa ratings ELO.
- Treina o modelo ELO com histórico.
- Gera previsões para jogos futuros e exibe resultados.

---

## Exemplo de Execução

```bash
python sassamaru-25.py
```

Certifique-se de que os arquivos de dados estejam nos caminhos esperados e que todas as dependências estejam instaladas.
