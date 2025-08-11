# -*- coding: utf-8 -*-
"""
sassamaru-25.py

Previsor de resultados do Campeonato Brasileiro utilizando modelo híbrido Poisson + ELO,
com ajuste por forma recente e ponderação de resultados baseada na diferença de ELO.

Principais funcionalidades:
- Cálculo de forças de ataque/defesa dos times via Poisson.
- Atualização dinâmica dos ratings ELO dos times, considerando vantagem de casa, diferença de gols, posição na tabela e peso pela diferença de ELO.
- Ajuste das probabilidades de resultado conforme a forma recente dos times.
- Geração de previsões para partidas futuras, exibindo probabilidades, palpites, ELOs e forma recente.

Dependências: pandas, numpy, scipy, matplotlib, scikit-learn

Autor: Paulo "mtgr18977"
Data: 2025-08-11
"""

import pandas as pd
import math
from datetime import datetime
from scipy.stats import skellam
import numpy as np
from sklearn.metrics import log_loss, accuracy_score, brier_score_loss, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# --- PARÂMETROS OTIMIZADOS ---
ELO_RATING_INICIAL = 1500
ELO_K_FACTOR_BASE = 30
ELO_VANTAGEM_CASA_PADRAO = 30
POISSON_MAX_GOLS = 8
ELO_INFLUENCE = 0.10
RECENT_FORM_GAMES = 5 # Number of recent games to consider for form
FORM_ADJUSTMENT_FACTOR = 0.05 # How much recent form influences probabilities

# --- FUNÇÕES DO MODELO ---

def calcular_forcas_poisson(df):
    """
    Calcula as forças de ataque e defesa (casa/fora) de cada time usando médias de gols.

    Parâmetros:
        df (pd.DataFrame): DataFrame com colunas 'mandante', 'visitante', 'gols_mandante', 'gols_visitante'

    Retorna:
        forcas (dict): Forças de ataque/defesa para cada time
        medias_liga (dict): Médias de gols da liga (casa/fora)
    """
    times = pd.unique(pd.concat([df['mandante'], df['visitante']]))
    forcas = {}
    media_gols_casa = df['gols_mandante'].mean()
    media_gols_fora = df['gols_visitante'].mean()
    medias_liga = {'gols_casa': media_gols_casa, 'gols_fora': media_gols_fora}

    for time in times:
        jogos_casa = df[df['mandante'] == time]
        jogos_fora = df[df['visitante'] == time]

        mean_gols_marcados_casa = jogos_casa['gols_mandante'].mean() if not jogos_casa.empty else 0
        mean_gols_sofridos_casa = jogos_casa['gols_visitante'].mean() if not jogos_casa.empty else 0
        mean_gols_marcados_fora = jogos_fora['gols_visitante'].mean() if not jogos_fora.empty else 0
        mean_gols_sofridos_fora = jogos_fora['gols_mandante'].mean() if not jogos_fora.empty else 0

        ataque_casa = (mean_gols_marcados_casa / media_gols_casa) if media_gols_casa > 0 else 1.0
        defesa_casa = (mean_gols_sofridos_casa / media_gols_fora) if media_gols_fora > 0 else 1.0
        ataque_fora = (mean_gols_marcados_fora / media_gols_fora) if media_gols_fora > 0 else 1.0
        defesa_fora = (mean_gols_sofridos_fora / media_gols_casa) if media_gols_casa > 0 else 1.0

        forcas[time] = {
            'ataque_casa': ataque_casa,
            'defesa_casa': defesa_casa,
            'ataque_fora': ataque_fora,
            'defesa_fora': defesa_fora
        }
    return forcas, medias_liga

def calcular_vantagens_casa(df):
    """
    Calcula a vantagem de jogar em casa para cada time, baseada no saldo médio de gols como mandante.

    Parâmetros:
        df (pd.DataFrame): DataFrame de partidas

    Retorna:
        vantagens (dict): Vantagem de casa para cada time
    """
    vantagens = {}
    times = pd.unique(df['mandante'])
    for time in times:
        jogos_casa = df[df['mandante'] == time]
        if not jogos_casa.empty:
            saldo_de_gols_medio = (jogos_casa['gols_mandante'] - jogos_casa['gols_visitante']).mean()
            if not np.isnan(saldo_de_gols_medio) and saldo_de_gols_medio > 0:
                 vantagem_rating = 60 * (saldo_de_gols_medio ** 0.8)
            else:
                vantagem_rating = 0
        else:
            vantagem_rating = 0
        vantagens[time] = vantagem_rating
    return vantagens

def calcular_forma_recente(df, n_jogos=5):
    """
    Calcula a forma recente (vitórias, empates, derrotas) dos times nos últimos n_jogos.

    Parâmetros:
        df (pd.DataFrame): DataFrame de partidas
        n_jogos (int): Número de jogos recentes a considerar

    Retorna:
        forma_recente (dict): Dicionário com contagem de resultados recentes para cada time
    """
    forma_recente = {}
    all_teams = pd.unique(pd.concat([df['mandante'], df['visitante']]))

    for team in all_teams:
        team_games = df[(df['mandante'] == team) | (df['visitante'] == team)]
        recent_games = team_games.tail(n_jogos)

        if len(recent_games) < n_jogos:
            forma_recente[team] = None
            continue

        wins = 0
        draws = 0
        losses = 0

        for _, game in recent_games.iterrows():
            if game['mandante'] == team:
                if game['gols_mandante'] > game['gols_visitante']:
                    wins += 1
                elif game['gols_mandante'] == game['gols_visitante']:
                    draws += 1
                else:
                    losses += 1
            else:
                if game['gols_visitante'] > game['gols_mandante']:
                    wins += 1
                elif game['gols_visitante'] == game['gols_mandante']:
                    draws += 1
                else:
                    losses += 1

        forma_recente[team] = {'vitorias': wins, 'empates': draws, 'derrotas': losses}

    return forma_recente

def poisson(lmbda, k):
    """
    Calcula a probabilidade de marcar k gols dado lambda (média esperada) usando a distribuição de Poisson.

    Parâmetros:
        lmbda (float): Média esperada de gols
        k (int): Número de gols

    Retorna:
        float: Probabilidade de marcar k gols
    """
    if lmbda < 1e-9: return 0.0
    if k > 30: return 0.0
    try:
        return math.exp(-lmbda) * (lmbda ** k) / math.factorial(k)
    except OverflowError:
        return 0.0


def prever_partida_hibrido(time_casa, time_visitante, context):
    """
    Calcula as probabilidades de vitória, empate e derrota para uma partida usando modelo híbrido Poisson + ELO.

    Parâmetros:
        time_casa (str): Nome do time mandante
        time_visitante (str): Nome do time visitante
        context (dict): Contexto com ratings ELO, forças Poisson, médias da liga e vantagens de casa

    Retorna:
        dict: Probabilidades, gols esperados, ELOs e palpite para a partida
    """
    elo_ratings = context['elo_ratings']
    forcas_poisson = context['forcas_poisson']
    medias_liga = context['medias_liga']
    vantagens_casa = context['vantagens_casa']

    if time_casa not in forcas_poisson or time_visitante not in forcas_poisson:
         return {
            'Mandante': time_casa.title(),
            'Visitante': time_visitante.title(),
            'Elo Mandante': ELO_RATING_INICIAL,
            'Elo Visitante': ELO_RATING_INICIAL,
            'Gols Esp. Mandante': medias_liga.get('gols_casa', 1.0),
            'Gols Esp. Visitante': medias_liga.get('gols_fora', 1.0),
            'P(Mandante)%': f"{1/3*100:.1f}",
            'P(Empate)%': f"{1/3*100:.1f}",
            'P(Visitante)%': f"{1/3*100:.1f}",
            'Palpite': 'Empate'
         }

    ataque_casa = forcas_poisson[time_casa]['ataque_casa']
    defesa_visitante = forcas_poisson[time_visitante]['defesa_fora']
    gols_base_casa = ataque_casa * defesa_visitante * medias_liga['gols_casa']
    ataque_visitante = forcas_poisson[time_visitante]['ataque_fora']
    defesa_casa = forcas_poisson[time_casa]['defesa_casa']
    gols_base_visitante = ataque_visitante * defesa_casa * medias_liga['gols_fora']
    rating_casa = elo_ratings.get(time_casa, ELO_RATING_INICIAL)
    rating_visitante = elo_ratings.get(time_visitante, ELO_RATING_INICIAL)
    vantagem_time_casa = vantagens_casa.get(time_casa, ELO_VANTAGEM_CASA_PADRAO)
    elo_diff = (rating_casa + vantagem_time_casa) - rating_visitante
    fator_ajuste_casa = 1 + (elo_diff / 1000) * ELO_INFLUENCE
    fator_ajuste_visitante = 1 - (elo_diff / 1000) * ELO_INFLUENCE
    gols_finais_casa = gols_base_casa * max(0.1, fator_ajuste_casa)
    gols_finais_visitante = gols_base_visitante * max(0.1, fator_ajuste_visitante)

    mu1, mu2 = gols_finais_casa, gols_finais_visitante
    if mu1 < 0 or mu2 < 0:
        prob_vitoria_casa, prob_empate, prob_vitoria_visitante = 1/3, 1/3, 1/3
    else:
        prob_vitoria_casa = 1 - skellam.cdf(0, mu1, mu2)
        prob_empate = skellam.pmf(0, mu1, mu2)
        prob_vitoria_visitante = skellam.cdf(-1, mu1, mu2)

    total_prob = prob_vitoria_casa + prob_empate + prob_vitoria_visitante
    if total_prob == 0:
        prob_vitoria_casa, prob_empate, prob_vitoria_visitante = 1/3, 1/3, 1/3
    else:
        prob_vitoria_casa /= total_prob
        prob_empate /= total_prob
        prob_vitoria_visitante /= total_prob

    probs = [prob_vitoria_casa, prob_empate, prob_vitoria_visitante]
    idx_max = probs.index(max(probs))
    palpite = ['Mandante', 'Empate', 'Visitante'][idx_max]

    return {
        'Mandante': time_casa.title(),
        'Visitante': time_visitante.title(),
        'Elo Mandante': int(rating_casa),
        'Elo Visitante': int(rating_visitante),
        'Gols Esp. Mandante': round(gols_finais_casa, 2),
        'Gols Esp. Visitante': round(gols_finais_visitante, 2),
        'P(Mandante)%': f"{prob_vitoria_casa * 100:.1f}",
        'P(Empate)%': f"{prob_empate * 100:.1f}",
        'P(Visitante)%': f"{prob_vitoria_visitante * 100:.1f}",
        'Palpite': palpite
    }

def prever_partida_hibrido_com_forma(time_casa, time_visitante, context, forma_recente):
    """
    Ajusta as probabilidades do modelo híbrido considerando a forma recente dos times.

    Parâmetros:
        time_casa (str): Nome do time mandante
        time_visitante (str): Nome do time visitante
        context (dict): Contexto do modelo
        forma_recente (dict): Forma recente dos times

    Retorna:
        dict: Probabilidades ajustadas, ELOs, gols esperados, forma recente e palpite
    """
    base_prediction_probs = prever_partida_hibrido(time_casa, time_visitante, context)

    prob_vc = float(base_prediction_probs['P(Mandante)%'].replace('%', '')) / 100
    prob_e = float(base_prediction_probs['P(Empate)%'].replace('%', '')) / 100
    prob_vv = float(base_prediction_probs['P(Visitante)%'].replace('%', '')) / 100

    forma_casa = forma_recente.get(time_casa)
    forma_visitante = forma_recente.get(time_visitante)

    if forma_casa is not None and forma_visitante is not None:
        total_casa_games = forma_casa['vitorias'] + forma_casa['empates'] + forma_casa['derrotas']
        total_visitante_games = forma_visitante['vitorias'] + forma_visitante['empates'] + forma_visitante['derrotas']

        if total_casa_games > 0 and total_visitante_games > 0:
            casa_win_pct = forma_casa['vitorias'] / total_casa_games
            visitante_win_pct = forma_visitante['vitorias'] / total_visitante_games

            diff_pct = casa_win_pct - visitante_win_pct

            prob_vc += diff_pct * FORM_ADJUSTMENT_FACTOR
            prob_vv -= diff_pct * FORM_ADJUSTMENT_FACTOR

    prob_vc = np.clip(prob_vc, 0, 1)
    prob_e = np.clip(prob_e, 0, 1)
    prob_vv = np.clip(prob_vv, 0, 1)

    total_prob = prob_vc + prob_e + prob_vv
    if total_prob > 0:
        prob_vc /= total_prob
        prob_e /= total_prob
        prob_vv /= total_prob
    else:
        prob_vc, prob_e, prob_vv = 1/3, 1/3, 1/3

    adjusted_probs = [prob_vc, prob_e, prob_vv]
    idx_max = adjusted_probs.index(max(adjusted_probs))
    adjusted_palpite = ['Mandante', 'Empate', 'Visitante'][idx_max]

    adjusted_prediction = base_prediction_probs.copy()
    adjusted_prediction['P(Mandante)%'] = f"{prob_vc * 100:.1f}"
    adjusted_prediction['P(Empate)%'] = f"{prob_e * 100:.1f}"
    adjusted_prediction['P(Visitante)%'] = f"{prob_vv * 100:.1f}"
    adjusted_prediction['Palpite'] = adjusted_palpite
    adjusted_prediction['Forma Mandante'] = forma_casa
    adjusted_prediction['Forma Visitante'] = forma_visitante

    return adjusted_prediction

def calcular_peso_elo(rating_vencedor, rating_perdedor):
    """
    Calcula o peso de uma vitória baseada na diferença de ELO.
    Vitórias improváveis (ELO menor vence ELO maior) têm peso maior.
    Retorna um fator multiplicativo para o K.

    Parâmetros:
        rating_vencedor (float): ELO do time vencedor
        rating_perdedor (float): ELO do time perdedor

    Retorna:
        float: Fator de peso para multiplicar o K
    """
    diff = rating_vencedor - rating_perdedor
    # Peso mínimo 0.7, máximo 1.3, centrado em 1.0 para diferença zero
    # Ajuste a função conforme desejado (exponencial, logística, etc.)
    peso = 1.0
    if diff < 0:
        # Vencedor tinha ELO menor: aumenta peso
        peso = min(1.3, 1.0 + abs(diff) / 800)
    elif diff > 0:
        # Vencedor tinha ELO maior: reduz peso
        peso = max(0.7, 1.0 - abs(diff) / 800)
    return peso

def atualizar_ratings_elo(rating_c, rating_v, placar_c, placar_v, vantagem_c, df_standings, time_c, time_v):
    """
    Atualiza os ratings ELO dos times após uma partida, considerando vantagem de casa, diferença de gols,
    posição na tabela e peso pela diferença de ELO.

    Parâmetros:
        rating_c (float): ELO do mandante antes do jogo
        rating_v (float): ELO do visitante antes do jogo
        placar_c (int): Gols do mandante
        placar_v (int): Gols do visitante
        vantagem_c (float): Vantagem de casa do mandante
        df_standings (pd.DataFrame): DataFrame com posições dos times
        time_c (str): Nome do mandante
        time_v (str): Nome do visitante

    Retorna:
        tuple: (novo_rating_c, novo_rating_v) - Novos ratings ELO para mandante e visitante
    """
    if placar_c > placar_v:
        resultado_real = 1.0
        rating_vencedor = rating_c
        rating_perdedor = rating_v
    elif placar_c == placar_v:
        resultado_real = 0.5
        rating_vencedor = None
        rating_perdedor = None
    else:
        resultado_real = 0.0
        rating_vencedor = rating_v
        rating_perdedor = rating_c

    exp_c = 1 / (1 + 10 ** ((rating_v - (rating_c + vantagem_c)) / 400))

    k = ELO_K_FACTOR_BASE

    standing_casa = df_standings.set_index('Team').get(time_c, {}).get('Position', df_standings['Position'].max() + 1)
    standing_visitante = df_standings.set_index('Team').get(time_v, {}).get('Position', df_standings['Position'].max() + 1)

    diff_standings = standing_casa - standing_visitante

    if (resultado_real == 1.0 and diff_standings > 0) or (resultado_real == 0.0 and diff_standings < 0):
        k *= 1.2
    elif (resultado_real == 1.0 and diff_standings <= 0) or (resultado_real == 0.0 and diff_standings >= 0):
         k = max(ELO_K_FACTOR_BASE * 0.5, k * 0.8)

    if resultado_real == 0.5:
         k = ELO_K_FACTOR_BASE

    diff_gols = abs(placar_c - placar_v)
    k *= (1 + 0.5 * max(0, diff_gols - 1))

    # Aplica o peso ELO apenas para vitórias/derrotas, não para empates
    if rating_vencedor is not None and rating_perdedor is not None:
        k *= calcular_peso_elo(rating_vencedor, rating_perdedor)

    novo_rating_c = rating_c + k * (resultado_real - exp_c)
    novo_rating_v = rating_v + k * ((1 - resultado_real) - (1 - exp_c))

    return novo_rating_c, novo_rating_v


# --- EXECUÇÃO PRINCIPAL ---
def main():
    """
    Executa o fluxo principal:
    - Carrega dados do campeonato e classificação.
    - Calcula forças, vantagens, forma recente e inicializa ratings ELO.
    - Treina o modelo ELO com histórico.
    - Gera previsões para jogos futuros e exibe resultados.

    Não recebe parâmetros.
    """
    try:
        df = pd.read_csv('/content/campeonato-brasileiro-full.csv')
    except FileNotFoundError:
        print("ERRO: Arquivo 'campeonato-brasileiro-full.csv' não encontrado.")
        return

    if 'gols_mandante' not in df.columns or 'gols_visitante' not in df.columns or 'resultado' not in df.columns:
        print("ERRO: Colunas 'gols_mandante', 'gols_visitante' ou 'resultado' não encontradas no CSV.")
        return

    df.dropna(subset=['gols_mandante', 'gols_visitante', 'resultado'], inplace=True)

    df['gols_mandante'] = df['gols_mandante'].astype(int)
    df['gols_visitante'] = df['gols_visitante'].astype(int)

    df['mandante'] = df['mandante'].str.strip().str.lower()
    df['visitante'] = df['visitante'].str.strip().str.lower()

    # Map 'resultado' column to standard labels
    outcome_mapping = {}
    for index, row in df.iterrows():
        mandante = row['mandante']
        visitante = row['visitante']
        resultado_raw = row['resultado'].strip().lower()

        if resultado_raw == 'empate':
            outcome_mapping[resultado_raw] = 'Empate'
        elif resultado_raw == mandante:
             outcome_mapping[resultado_raw] = 'Mandante'
        elif resultado_raw == visitante:
             outcome_mapping[resultado_raw] = 'Visitante'
        elif resultado_raw == 'vitoria_mandante':
             outcome_mapping[resultado_raw] = 'Mandante'
        elif resultado_raw == 'vitoria_visitante':
             outcome_mapping[resultado_raw] = 'Visitante'

    df['resultado_mapped'] = df['resultado'].str.strip().str.lower().map(outcome_mapping).fillna('Unknown_Outcome')
    df['resultado'] = df['resultado_mapped']
    df.drop(columns=['resultado_mapped'], inplace=True)


    # Load and process standings data
    file_path = '/content/classificacao.txt'
    standings_data = []
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()

        for line in lines:
            line = line.strip().lower()
            if not line: continue

            parts = line.split(',')
            if len(parts) == 3:
                try:
                    team_name = parts[0].strip()
                    position = int(parts[1].strip())
                    points = int(parts[2].strip())
                    standings_data.append({'Position': position, 'Team': team_name, 'Points': points})
                    continue
                except ValueError:
                    pass

            try:
                first_space_index = line.find(' ')
                if first_space_index == -1: continue
                position_str = line[:first_space_index].replace('.', '')
                position = int(position_str)
                team_and_points = line[first_space_index:].strip()
                last_space_index = team_and_points.rfind(' ')
                if last_space_index == -1: continue
                team_name = team_and_points[:last_space_index].strip()
                points_str = team_and_points[last_space_index:].strip()
                points = int(points_str)
                standings_data.append({'Position': position, 'Team': team_name, 'Points': points})
            except ValueError:
                print(f"Skipping malformed line in standings: {line}")
                continue
        df_standings = pd.DataFrame(standings_data)
        df_standings['Team'] = df_standings['Team'].str.lower()

    except FileNotFoundError:
        print(f"ERRO: Arquivo '{file_path}' não encontrado.")
        return
    except Exception as e:
        print(f"Erro ao processar arquivo de classificação: {e}")
        return

    # Calculate Poisson forces and league averages
    forcas_poisson, medias_liga = calcular_forcas_poisson(df)

    # Calculate home advantage ratings
    vantagens_casa = calcular_vantagens_casa(df)

    # Calculate recent form data
    recent_form_data = calcular_forma_recente(df, n_jogos=RECENT_FORM_GAMES)

    # Initialize Elo ratings
    elo_ratings = {}
    all_teams = pd.unique(pd.concat([df['mandante'], df['visitante']]))
    for team in all_teams:
        elo_ratings[team] = ELO_RATING_INICIAL


    # Train Elo model by iterating through historical data
    print("Training Elo model with historical data and standings adjustment...")
    for index, partida in df.iterrows():
        time_c, time_v = partida['mandante'], partida['visitante']
        placar_c, placar_v = partida['gols_mandante'], partida['gols_visitante']

        rating_c = elo_ratings.get(time_c, ELO_RATING_INICIAL)
        rating_v = elo_ratings.get(time_v, ELO_RATING_INICIAL)

        vantagem_c = vantagens_casa.get(time_c, ELO_VANTAGEM_CASA_PADRAO)

        novo_rating_c, novo_rating_v = atualizar_ratings_elo(rating_c, rating_v, placar_c, placar_v, vantagem_c, df_standings, time_c, time_v)

        elo_ratings[time_c] = novo_rating_c
        elo_ratings[time_v] = novo_rating_v
    print("Elo training complete.")

    # Prepare context for prediction
    context = {
        'elo_ratings': elo_ratings,
        'forcas_poisson': forcas_poisson,
        'medias_liga': medias_liga,
        'vantagens_casa': vantagens_casa
    }

    # Define the list of games for prediction
    jogos_para_prever = [
        ('fluminense', 'palmeiras'),
        ('ceara', 'mirassol'),
        ('corinthians', 'cruzeiro'),
        ('santos', 'internacional'),
        ('bragantino', 'flamengo'),
        ('vitoria', 'sport'),
        ('juventude', 'sao paulo'),
        ('vasco', 'bahia'),
        ('atletico mineiro', 'fortaleza'),
        ('gremio', 'botafogo')
    ]

    # Generate predictions for the specified games
    resultados_rodada_com_forma = []
    for casa, visita in jogos_para_prever:
        try:
            pred_com_forma = prever_partida_hibrido_com_forma(casa, visita, context, recent_form_data)
            resultados_rodada_com_forma.append(pred_com_forma)
        except Exception as e:
            print(f"Erro ao prever (com forma) {casa} x {visita}: {e}")
            # Optionally append a default prediction or an error indicator
            resultados_rodada_com_forma.append({
                'Mandante': casa.title(),
                'Visitante': visita.title(),
                'Elo Mandante': 'N/A',
                'Elo Visitante': 'N/A',
                'Gols Esp. Mandante': 'N/A',
                'Gols Esp. Visitante': 'N/A',
                'P(Mandante)%': 'N/A',
                'P(Empate)%': 'N/A',
                'P(Visitante)%': 'N/A',
                'Palpite': 'Erro',
                'Forma Mandante': 'N/A',
                'Forma Visitante': 'N/A'
            })

    # Create a pandas DataFrame from the results
    df_proxima_rodada = pd.DataFrame(resultados_rodada_com_forma)

    # Print the predictions in markdown format
    print("\n# Previsões para a Próxima Rodada (Modelo Híbrido com Forma Recente)\n")

    # Define columns to display and their order
    cols_to_display = [
        'Mandante', 'Visitante',
        'P(Mandante)%', 'P(Empate)%', 'P(Visitante)%',
        'Palpite',
        'Elo Mandante', 'Elo Visitante', # Include Elo ratings
        'Gols Esp. Mandante', 'Gols Esp. Visitante', # Include Expected Goals
        'Forma Mandante', 'Forma Visitante' # Include recent form details
    ]

    # Ensure columns exist before selecting
    cols_to_display = [col for col in cols_to_display if col in df_proxima_rodada.columns]

    print(df_proxima_rodada[cols_to_display].to_markdown(index=False))

if __name__ == "__main__":
    main()
