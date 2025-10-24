Script para verificar os resultados dos modelos já treinados
============================================================

Este script carrega os modelos salvos e mostra suas métricas de performance.

import numpy as np
import pandas as pd
from pathlib import Path
from poker_ai.cluster import PokerHandClusterer
from poker_ai.data import load_poker_dataset

def verificar_modelo(modelo_path, dados_path, nome_modelo):
    print(f"\n{'='*60}")
    print(f"🤖 VERIFICANDO: {nome_modelo}")
    print(f"{'='*60}")
    
    try:
        modelo = PokerHandClusterer.load_model(modelo_path)
        print(f"✅ Modelo carregado: {modelo_path}")
        
        dados = load_poker_dataset(
            dados_path,
            label_column="tipo_mao",
            drop_columns=None
        )
        print(f"✅ Dados carregados: {len(dados.features)} amostras")
        
        print(f"\n📊 Calculando métricas...")
        metricas = modelo.evaluate(
            dados.features,
            labels=dados.labels
        )
        
        print(f"\n📈 RESULTADOS:")
        print(f"   🔹 Inércia: {metricas.inertia:.2f}")
        if metricas.silhouette is not None:
            print(f"   🔹 Silhouette Score: {metricas.silhouette:.4f}")
        if metricas.accuracy is not None:
            print(f"   🔹 Acurácia: {metricas.accuracy:.2%}")
        
        print(f"\n🔧 CONFIGURAÇÃO:")
        print(f"   🔹 Número de clusters: {modelo.n_clusters}")
        print(f"   🔹 Random state: {modelo.random_state}")
        print(f"   🔹 Tem mapeamento de labels: {'Sim' if modelo.cluster_label_map else 'Não'}")
        
        print(f"\n🔮 TESTE DE PREDIÇÃO:")
        maos_teste = np.array([
            [0, 2, 0, 3, 0, 4, 0, 5, 0, 6],
            [0, 10, 1, 10, 2, 10, 3, 5, 0, 5],
            [0, 7, 1, 7, 2, 2, 3, 3, 0, 4],
        ])
        
        clusters = modelo.predict(maos_teste)
        print(f"   🔹 Clusters preditos: {clusters}")
        
        if modelo.cluster_label_map:
            labels = modelo.predict_labels(maos_teste)
            print(f"   🔹 Labels preditos: {labels}")
            
            tipos_mao = {
                0: "Sem par", 1: "Um par", 2: "Dois pares", 3: "Trinca",
                4: "Straight", 5: "Flush", 6: "Full house", 7: "Quadra",
                8: "Straight flush", 9: "Royal flush"
            }
            
            print(f"   🔹 Tipos de mão:")
            for i, (cluster, label) in enumerate(zip(clusters, labels)):
                tipo = tipos_mao.get(label, f"Desconhecido ({label})")
                print(f"      Mão {i+1}: {tipo} (Cluster {cluster})")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro ao verificar {nome_modelo}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🃏 VERIFICADOR DE RESULTADOS DOS MODELOS 🃏")
    print("="*60)
    
    pasta_resultados = Path("resultados")
    
    modelos = [
        {
            "modelo_path": pasta_resultados / "modelo_100_porcento.pkl",
            "dados_path": pasta_resultados / "poker_dataset_convertido.csv",
            "nome_modelo": "Modelo 100% (Todos os dados)"
        },
        {
            "modelo_path": pasta_resultados / "modelo_split_70_30.pkl", 
            "dados_path": pasta_resultados / "dataset_treino_70.csv",
            "nome_modelo": "Modelo Split 70/30 (Treino)"
        }
    ]
    
    if not pasta_resultados.exists():
        print(f"❌ Pasta de resultados não encontrada: {pasta_resultados}")
        print("Execute primeiro o script de treinamento!")
        return
    
    print(f"📁 Pasta de resultados: {pasta_resultados.absolute()}")
    
    print(f"\n📄 Arquivos encontrados:")
    for arquivo in pasta_resultados.iterdir():
        tamanho = arquivo.stat().st_size / (1024*1024)
        print(f"   🔹 {arquivo.name} ({tamanho:.1f} MB)")
    
    sucessos = 0
    for modelo_info in modelos:
        if verificar_modelo(**modelo_info):
            sucessos += 1
    
    print(f"\n{'='*60}")
    print(f"✅ VERIFICAÇÃO CONCLUÍDA: {sucessos}/{len(modelos)} modelos OK")
    print(f"{'='*60}")
    
    if sucessos > 0:
        print(f"\n💡 DICAS:")
        print(f"   🔹 Inércia menor = clusters mais compactos")
        print(f"   🔹 Silhouette próximo de 1 = clusters bem separados")
        print(f"   🔹 Acurácia alta = predições corretas")
        print(f"\n🎉 Seus modelos estão funcionando perfeitamente! 🃏✨")

if __name__ == "__main__":
    main()
