import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Módulos de Scikit-Learn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    roc_curve,
    roc_auc_score
)

# Conector personalizado a la base de datos
from src.database.connection2 import MySQLConnector

def main():
    print("=================================================================")
    print("  ECOSISTEMA ANALÍTICO END-TO-END: ENTRENAMIENTO Y DEPLOYMENT")
    print("=================================================================\n")

    # -------------------------------------------------------------
    # FASE 1: INGESTA, PREPROCESAMIENTO Y FEATURE ENGINEERING
    # -------------------------------------------------------------
    db = MySQLConnector()

    query = """
        SELECT 
            p.player_id,
            p.name AS jugador,
            p.sub_position AS posicion,
            p.height_in_cm AS estatura_cm,
            TIMESTAMPDIFF(YEAR, p.date_of_birth, CURDATE()) AS edad,
            p.market_value_in_eur AS valor_mercado
        FROM players p
        WHERE p.market_value_in_eur IS NOT NULL
          AND p.date_of_birth IS NOT NULL;
    """

    print("[1] Extrayendo datos desde MySQL...")
    df = db.extract_data(query)

    df['estatura_cm'] = pd.to_numeric(df['estatura_cm'], errors='coerce')
    df['edad'] = pd.to_numeric(df['edad'], errors='coerce')
    df['valor_mercado'] = pd.to_numeric(df['valor_mercado'], errors='coerce')
    df['posicion'] = df['posicion'].fillna('Sin Posicion')

    # Guardar las medianas de imputación para reutilizar en producción
    medianas_estatura = df.groupby('posicion')['estatura_cm'].median().to_dict()
    mediana_global_estatura = df['estatura_cm'].median()
    mediana_global_edad = df['edad'].median()

    # Imputación por Mediana Agrupada
    df['estatura_cm'] = df.groupby('posicion')['estatura_cm'].transform(lambda x: x.fillna(x.median()))
    df['estatura_cm'] = df['estatura_cm'].fillna(mediana_global_estatura)
    df['edad'] = df['edad'].fillna(mediana_global_edad)

    # Target Binario por Percentil: Élite (Top 25% = 1, Resto 75% = 0)
    umbral_elite = df['valor_mercado'].quantile(0.75)
    df['es_elite'] = (df['valor_mercado'] >= umbral_elite).astype(int)

    # One-Hot Encoding: Matriz de variables (X) y vector objetivo (y)
    df_dummies = pd.get_dummies(df[['posicion']], drop_first=True, dtype=int)
    X = pd.concat([df[['edad', 'estatura_cm']], df_dummies], axis=1)
    y = df['es_elite']

    # Depuración Anti-NaN
    filas_con_nan = X.isna().any(axis=1)
    if filas_con_nan.sum() > 0:
        X = X[~filas_con_nan]
        y = y[~filas_con_nan]

    # Muestreo Estratificado: División 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"[OK] Preprocesamiento completado. Registros procesados: {X.shape[0]}.")
    print(f"     Clase 0 (Estándar): {np.sum(y==0)} | Clase 1 (Élite): {np.sum(y==1)}\n")

    # -------------------------------------------------------------
    # FASE 2A: DIAGNÓSTICO DE OVERFITTING (SIN PODA vs PODA PREVIA)
    # -------------------------------------------------------------
    print("[2] Diagnóstico de Overfitting: Árbol sin poda vs Árbol con poda previa...")
    tree_libre = DecisionTreeClassifier(random_state=42)
    tree_libre.fit(X_train, y_train)

    tree_podado = DecisionTreeClassifier(max_depth=3, random_state=42)
    tree_podado.fit(X_train, y_train)

    tabla = pd.DataFrame({
        'Modelo': ['Árbol sin poda', 'Árbol con poda (max_depth=3)'],
        'Accuracy Train': [accuracy_score(y_train, tree_libre.predict(X_train)),
                           accuracy_score(y_train, tree_podado.predict(X_train))],
        'Accuracy Test': [accuracy_score(y_test, tree_libre.predict(X_test)),
                          accuracy_score(y_test, tree_podado.predict(X_test))]
    })
    tabla['Brecha (Train - Test)'] = tabla['Accuracy Train'] - tabla['Accuracy Test']

    print("\n=================================================")
    print("  DIAGNÓSTICO DE OVERFITTING (ACCURACY TRAIN vs TEST)")
    print("=================================================")
    print(tabla.round(4).to_string(index=False))
    print("=================================================\n")

    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(2)
    ancho = 0.35
    barras_train = ax.bar(x - ancho/2, tabla['Accuracy Train'], ancho, color='#2980b9', edgecolor='black', alpha=0.85, label='Train')
    barras_test = ax.bar(x + ancho/2, tabla['Accuracy Test'], ancho, color='#e67e22', edgecolor='black', alpha=0.85, label='Test')
    for barras in (barras_train, barras_test):
        for b in barras:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01, f'{b.get_height():.3f}', ha='center', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(['Árbol sin poda', 'Árbol con poda'])
    ax.set_ylim(0, 1.1)
    ax.set_title('Diagnóstico de Overfitting: Accuracy Train vs Test', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6, axis='y')
    ax.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    plt.savefig('diagnostico_overfitting.png', dpi=300)
    print("[OK] Gráfica guardada como 'diagnostico_overfitting.png'.\n")
    plt.close(fig)

    # -------------------------------------------------------------
    # FASE 2B: PARADOJA DE LA EXACTITUD Y class_weight='balanced'
    # -------------------------------------------------------------
    print("[3] Paradoja de la Exactitud: Árbol sin tratamiento de desbalance...")
    tree_base = DecisionTreeClassifier(max_depth=3, random_state=42)
    tree_base.fit(X_train, y_train)
    y_pred_base = tree_base.predict(X_test)

    print("\n--- MODELO BASE (DESBALANCEADO) ---")
    print(f"  • Accuracy Global: {accuracy_score(y_test, y_pred_base):.4f}")
    print(f"  • F1-Score (Clase Élite): {f1_score(y_test, y_pred_base, pos_label=1):.4f}")
    print("\nReporte Detallado:")
    print(classification_report(y_test, y_pred_base, target_names=['Estándar(0)', 'Élite(1)']))

    print("\n[4] Aplicando penalización de costo (class_weight='balanced')...")
    tree_bal = DecisionTreeClassifier(max_depth=3, class_weight='balanced', random_state=42)
    tree_bal.fit(X_train, y_train)
    y_pred_bal = tree_bal.predict(X_test)

    print("\n--- MODELO CON PESOS BALANCEADOS ---")
    print(f"  • Accuracy Global: {accuracy_score(y_test, y_pred_bal):.4f}")
    print(f"  • F1-Score (Clase Élite): {f1_score(y_test, y_pred_bal, pos_label=1):.4f}")
    print("\nReporte Detallado:")
    print(classification_report(y_test, y_pred_bal, target_names=['Estándar(0)', 'Élite(1)']))

    # -------------------------------------------------------------
    # FASE 2C: OPTIMIZACIÓN DEL RANDOM FOREST CON GRIDSEARCHCV
    # -------------------------------------------------------------
    print("\n[5] Ejecutando GridSearchCV sobre Random Forest (Búsqueda sistemática + K-Fold CV)...")

    param_grid = {
        'max_depth': [2, 3, 4, 5, 6],
        'min_samples_leaf': [3, 5, 10, 15],
        'class_weight': ['balanced', None]
    }

    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(n_estimators=100, random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    print(f"\nMejores Hiperparámetros Encontrados: {grid_search.best_params_}")
    print(f"Mejor F1-Score Promedio en CV: {grid_search.best_score_:.4f}")

    rf_model = grid_search.best_estimator_
    y_pred_rf = rf_model.predict(X_test)

    print("\n=================================================")
    print("  EVALUACIÓN FINAL DEL RANDOM FOREST ÓPTIMO (TEST SET)")
    print("=================================================")
    print(classification_report(y_test, y_pred_rf, target_names=['Estándar (0)', 'Élite (1)']))
    print("Matriz de Confusión:")
    print(confusion_matrix(y_test, y_pred_rf))
    print("=================================================\n")

    # -------------------------------------------------------------
    # FASE 3: EVALUACIÓN DE PROBABILIDADES Y CURVA ROC-AUC
    # -------------------------------------------------------------
    print("[6] Árbol de Decisión Individual vs Random Forest (ROC-AUC)...")
    tree_model = DecisionTreeClassifier(max_depth=4, class_weight='balanced', random_state=42)
    tree_model.fit(X_train, y_train)

    y_prob_tree = tree_model.predict_proba(X_test)[:, 1]
    y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

    auc_tree = roc_auc_score(y_test, y_prob_tree)
    auc_rf = roc_auc_score(y_test, y_prob_rf)

    print("\n=================================================")
    print("  COMPARATIVA ROC-AUC (CAPACIDAD DE DISCRIMINACIÓN)")
    print("=================================================")
    print(f"  • Árbol de Decisión Único ROC-AUC: {auc_tree:.4f}")
    print(f"  • Random Forest Classifier ROC-AUC: {auc_rf:.4f}")
    print("=================================================\n")

    importancias = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)

    print("Top Variables Más Relevantes según Random Forest:")
    print(importancias.head(5))
    print("\n")

    print("[7] Generando panel visual de resultados...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    fpr_tree, tpr_tree, _ = roc_curve(y_test, y_prob_tree)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

    axes[0].plot(fpr_tree, tpr_tree, color='#e67e22', linestyle='--', linewidth=2,
                 label=f'Árbol Único (AUC = {auc_tree:.3f})')
    axes[0].plot(fpr_rf, tpr_rf, color='#27ae60', linewidth=2.5,
                 label=f'Random Forest (AUC = {auc_rf:.3f})')
    axes[0].plot([0, 1], [0, 1], color='gray', linestyle=':', label='Clasificador Aleatorio (AUC = 0.500)')
    axes[0].set_title('Comparativa de Curvas ROC', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Tasa de Falsos Positivos (FPR - Falsas Alarmas)', fontsize=10)
    axes[0].set_ylabel('Tasa de Verdaderos Positivos (TPR - Recall)', fontsize=10)
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend(loc='lower right', frameon=True)

    top_importancias = importancias.head(6).sort_values(ascending=True)
    axes[1].barh(top_importancias.index, top_importancias.values, color='#2980b9', edgecolor='black', alpha=0.85)
    axes[1].set_title('Top Importancia de Variables (Random Forest)', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Importancia Relativa (Gini Impurity)', fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.6, axis='x')

    plt.tight_layout()
    plt.savefig('evaluacion_rf_vs_tree.png', dpi=300)
    print("[OK] Gráficas guardadas como 'evaluacion_rf_vs_tree.png'.\n")
    plt.close(fig)

    # -------------------------------------------------------------
    # FASE 4: SERIALIZACIÓN (JOBLIB) - PAQUETE DE DESPLIEGUE
    # -------------------------------------------------------------
    print("[8] Entrenando modelo final con los mejores hiperparámetros y exportando...")
    rf_final = RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1, **grid_search.best_params_
    )
    rf_final.fit(X, y)

    paquete_despliegue = {
        'modelo': rf_final,
        'columnas_X': X.columns.tolist(),
        'medianas_estatura': medianas_estatura,
        'mediana_global_estatura': mediana_global_estatura,
        'mediana_global_edad': mediana_global_edad,
        'umbral_elite_eur': umbral_elite
    }

    os.makedirs('models', exist_ok=True)
    ruta_modelo = 'models/modelo_scouting_rf.joblib'
    joblib.dump(paquete_despliegue, ruta_modelo)
    print(f"Paquete de modelo exportado exitosamente en: '{ruta_modelo}'")

if __name__ == "__main__":
    main()