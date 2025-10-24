Script de teste completo para o modelo de IA de Poker
======================================================

Este script demonstra como:
1. Treinar o modelo a partir de um CSV
2. Salvar o modelo treinado
3. Carregar o modelo salvo
4. Fazer predições com novas mãos de poker

Uso:
    python test_modelo_completo.py

import numpy as np
from pathlib import Path
from poker_ai.cluster import PokerHandClusterer
from poker_ai.data import load_poker_dataset

def criar_mao_teste():
    Cria algumas mãos de teste para demonstração.
    Formato: [naipe1, valor1, naipe2, valor2, ...]
    return np.array([
        [0, 2, 0, 3, 0, 4, 0, 5, 0, 6],
        [0, 10, 1, 10, 2, 10, 3, 5, 0, 5],
        [0, 7, 1, 7, 2, 2, 3, 3, 0, 4],
        [0, 4, 1, 6, 2, 8, 3, 10, 0, 12],
    ])

def main():
    print("=" * 60)
    print("🃏 TESTE COMPLETO DO MODELO DE IA DE POKER 🃏")
    print("=" * 60)
    
    csv_path = Path("exemplo_poker.csv")
    model_path = Path("modelo_poker_treinado.pkl")
    
    print("\n📂 PASSO 1: Carregando dados do CSV...")
    if not csv_path.exists():
        print(f"❌ ERRO: Arquivo {csv_path} não encontrado!")
        print("Execute primeiro o script para criar o CSV de exemplo.")
        return
    
    encoded = load_poker_dataset(
        csv_path,
        label_column="tipo_mao",
        drop_columns=None,
    )
    
    print(f"✅ Dados carregados com sucesso!")
    print(f"   - Total de amostras: {len(encoded.features)}")
    print(f"   - Número de features: {encoded.features.shape[1]}")
    print(f"   - Colunas: {', '.join(encoded.feature_columns)}")
    
    print("\n🤖 PASSO 2: Treinando o modelo K-Means...")
    
    n_clusters = len(np.unique(encoded.labels)) if encoded.labels is not None else 10
    print(f"   - Número de clusters: {n_clusters}")
    
    clusterer = PokerHandClusterer(
        n_clusters=n_clusters,
        random_state=42,
    )
    
    clusterer.fit(encoded.features)
    print("✅ Modelo treinado!")
    
    if encoded.labels is not None:
        clusterer.build_label_map(encoded.features, encoded.labels)
        print("✅ Mapeamento de clusters para rótulos construído!")
    
    print("\n📊 PASSO 3: Avaliando o modelo...")
    metrics = clusterer.evaluate(
        encoded.features,
        labels=encoded.labels,
    )
    
    print(f"   - Inércia: {metrics.inertia:.2f}")
    if metrics.silhouette is not None:
        print(f"   - Silhouette Score: {metrics.silhouette:.4f}")
    if metrics.accuracy is not None:
        print(f"   - Acurácia: {metrics.accuracy:.2%}")
    
    print(f"\n💾 PASSO 4: Salvando o modelo em {model_path}...")
    clusterer.save_model(model_path)
    print("✅ Modelo salvo com sucesso!")
    
    print(f"\n📥 PASSO 5: Carregando o modelo de {model_path}...")
    modelo_carregado = PokerHandClusterer.load_model(model_path)
    print("✅ Modelo carregado com sucesso!")
    
    print("\n🔮 PASSO 6: Fazendo predições com mãos de teste...")
    maos_teste = criar_mao_teste()
    
    clusters_preditos = modelo_carregado.predict(maos_teste)
    print(f"   - Clusters preditos: {clusters_preditos}")
    
    if modelo_carregado.cluster_label_map:
        labels_preditos = modelo_carregado.predict_labels(maos_teste)
        print(f"   - Rótulos preditos (tipo de mão): {labels_preditos}")
        
        tipos_mao = {
            0: "Sem par (Nothing)",
            1: "Um par (One pair)",
            2: "Dois pares (Two pairs)",
            3: "Trinca (Three of a kind)",
            4: "Straight",
            5: "Flush",
            6: "Full house",
            7: "Quadra (Four of a kind)",
            8: "Straight flush",
            9: "Royal flush"
        }
        
        print("\n   📋 Detalhes das predições:")
        for i, (label, cluster) in enumerate(zip(labels_preditos, clusters_preditos)):
            tipo = tipos_mao.get(label, f"Desconhecido ({label})")
            print(f"      Mão {i+1}: {tipo} (Cluster {cluster})")
    
    print("\n" + "=" * 60)
    print("✅ TESTE CONCLUÍDO COM SUCESSO! ✅")
    print("=" * 60)
    print("\n📝 Resumo:")
    print(f"   - CSV de entrada: {csv_path}")
    print(f"   - Modelo salvo em: {model_path}")
    print(f"   - Amostras treinadas: {len(encoded.features)}")
    print(f"   - Predições realizadas: {len(maos_teste)}")
    print("\n💡 O modelo está pronto para ser usado em produção!")

if __name__ == "__main__":
    main()
