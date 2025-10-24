═══════════════════════════════════════════════════════════════
    🃏 SCRIPT DE TREINAMENTO COMPLETO 🃏
═══════════════════════════════════════════════════════════════

Este script executa todo o pipeline:
1. Converte o dataset original para formato do modelo
2. Treina com 100% dos dados
3. Cria split 70/30 em CSVs separados
4. Treina com o split e avalia

Uso:
    python treinar_completo.py
    
    Ou com limite de linhas para teste:
    python treinar_completo.py --max-linhas 10000

import argparse
import sys
from pathlib import Path

from converter_dataset import converter_dataset

from poker_ai.cli import train_on_full_dataset, train_with_split

def executar_pipeline_completo(
    arquivo_original,
    max_linhas=None,
    pasta_saida="resultados"
):
    Executa o pipeline completo de conversão e treinamento.
    
    Args:
        arquivo_original: Caminho do poker_dataset_textified.csv
        max_linhas: Limite de linhas (None = todas)
        pasta_saida: Pasta onde salvar resultados
    print("=" * 70)
    print("🃏 PIPELINE COMPLETO DE TREINAMENTO 🃏")
    print("=" * 70)
    
    pasta = Path(pasta_saida)
    pasta.mkdir(exist_ok=True)
    print(f"\n📁 Pasta de resultados: {pasta.absolute()}")
    
    print("\n" + "=" * 70)
    print("📝 PASSO 1: CONVERTENDO DATASET")
    print("=" * 70)
    
    arquivo_convertido = pasta / "poker_dataset_convertido.csv"
    
    try:
        df_convertido = converter_dataset(
            arquivo_entrada=arquivo_original,
            arquivo_saida=arquivo_convertido,
            max_linhas=max_linhas,
            usar_river=True,
            mostrar_progresso=True
        )
        print(f"\n✅ Dataset convertido: {arquivo_convertido}")
    except Exception as e:
        print(f"\n❌ Erro na conversão: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 70)
    print("🤖 PASSO 2: TREINANDO COM 100% DOS DADOS")
    print("=" * 70)
    
    modelo_100 = pasta / "modelo_100_porcento.pkl"
    
    try:
        train_on_full_dataset(
            csv_path=arquivo_convertido,
            label_column="tipo_mao",
            drop_columns=None,
            n_clusters=None,
            random_state=42,
            model_output_path=modelo_100
        )
        print(f"\n✅ Modelo treinado com 100%: {modelo_100}")
    except Exception as e:
        print(f"\n❌ Erro no treinamento 100%: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 70)
    print("📊 PASSO 3: CRIANDO SPLIT 70/30 E TREINANDO")
    print("=" * 70)
    
    treino_csv = pasta / "dataset_treino_70.csv"
    teste_csv = pasta / "dataset_teste_30.csv"
    modelo_split = pasta / "modelo_split_70_30.pkl"
    
    import pandas as pd
    df_temp = pd.read_csv(arquivo_convertido)
    contagem_classes = df_temp['tipo_mao'].value_counts()
    pode_estratificar = contagem_classes.min() >= 2
    
    if pode_estratificar:
        print("✅ Usando estratificação para manter proporções de classes")
        stratify_col = "tipo_mao"
    else:
        print(f"⚠️  Desabilitando estratificação (classe com apenas {contagem_classes.min()} exemplo)")
        stratify_col = None
    
    try:
        train_with_split(
            csv_path=arquivo_convertido,
            label_column="tipo_mao",
            drop_columns=None,
            n_clusters=None,
            train_size=0.7,
            random_state=42,
            stratify_column=stratify_col,
            output_train_path=treino_csv,
            output_test_path=teste_csv,
            model_output_path=modelo_split
        )
        print(f"\n✅ CSV de treino (70%): {treino_csv}")
        print(f"✅ CSV de teste (30%): {teste_csv}")
        print(f"✅ Modelo com split: {modelo_split}")
    except Exception as e:
        print(f"\n❌ Erro no treinamento com split: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 70)
    print("✅ PIPELINE CONCLUÍDO COM SUCESSO! 🎉")
    print("=" * 70)
    
    print(f"\n📂 Arquivos gerados em: {pasta.absolute()}")
    print(f"\n📄 CSVs:")
    print(f"   ├── {arquivo_convertido.name} (dataset convertido)")
    print(f"   ├── {treino_csv.name} (70% treino)")
    print(f"   └── {teste_csv.name} (30% teste)")
    
    print(f"\n🤖 Modelos treinados:")
    print(f"   ├── {modelo_100.name} (treinado com 100%)")
    print(f"   └── {modelo_split.name} (treinado com split 70/30)")
    
    print(f"\n💡 Como usar os modelos:")
    print(f"   from poker_ai.cluster import PokerHandClusterer")
    print(f"   modelo = PokerHandClusterer.load_model('{modelo_split}')")
    print(f"   # ... fazer predições ...")
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline completo: converte dataset e treina modelos",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--arquivo",
        type=str,
        default=r"C:\Python_Projects\AI_Poker\poker_dataset_textified.csv",
        help="Caminho do arquivo poker_dataset_textified.csv",
    )
    
    parser.add_argument(
        "--max-linhas",
        type=int,
        default=None,
        help="Limite de linhas para processar (útil para testes rápidos)",
    )
    
    parser.add_argument(
        "--pasta-saida",
        type=str,
        default="resultados",
        help="Pasta onde salvar todos os resultados",
    )
    
    args = parser.parse_args()
    
    arquivo_path = Path(args.arquivo)
    if not arquivo_path.exists():
        print(f"❌ Erro: Arquivo não encontrado: {args.arquivo}")
        print(f"\nVerifique se o caminho está correto.")
        sys.exit(1)
    
    if args.max_linhas:
        print(f"\n⚠️  Processando apenas {args.max_linhas:,} linhas (modo teste)")
    else:
        print(f"\n⚠️  Processando arquivo completo (~1 milhão de linhas)")
        print(f"   Isso pode levar alguns minutos...")
        print(f"\n   💡 Dica: Use --max-linhas 10000 para teste rápido")
        
        resposta = input("\n   Continuar? (s/n): ").strip().lower()
        if resposta not in ['s', 'sim', 'y', 'yes']:
            print("\n❌ Operação cancelada pelo usuário")
            sys.exit(0)
    
    sucesso = executar_pipeline_completo(
        arquivo_original=args.arquivo,
        max_linhas=args.max_linhas,
        pasta_saida=args.pasta_saida
    )
    
    if sucesso:
        print("\n🎉 Tudo pronto! Seus modelos estão prontos para uso! 🃏✨")
        sys.exit(0)
    else:
        print("\n❌ Ocorreram erros durante o pipeline")
        sys.exit(1)

if __name__ == "__main__":
    main()
