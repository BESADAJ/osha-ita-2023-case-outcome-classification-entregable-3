# Decisión de la Fase A

Generado 2026-10-21 02:14. Regla: `regla_decision.md` (fijada antes de ver resultados).
Referencia: `xgboost__class_weight__bayes` (BA media por pliegue 0.505 en `master_registry.csv`).
Referencia, pliegues juntos: H = 8/12, F = 5337, R234 = 0.466, AUC = 0.704, LL = 1.107.

| Candidato | H | F | a1 | a2 | R234 | AUC | LL | (b) | Mejora |
|---|---|---|---|---|---|---|---|---|---|
| PT-bagging | 1/12 | 296 | no | no | 0.487 | 0.644 | 1.635 | no | **no** |
| MultiRandBal | 0/12 | 21 | no | no | 0.482 | 0.631 | 1.252 | no | **no** |
| Jerárquica | 0/12 | 0 | no | no | 0.495 | 0.684 | 1.214 | no | **no** |

Ningún candidato cumple la regla. Pasa a la Fase B el que más muertes acierta con menos casos marcados: **PT-bagging**; el informe dirá que **no mejoró**.

Sesgo declarado: elegir el mejor de 3 candidatos sobrestima su ventaja.
