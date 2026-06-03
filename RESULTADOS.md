# Resultados de las simulaciones — LunarLander-v3

Comparativa entre un agente **Q-Learning** (Q-table con discretización) y un
agente **Deep Q-Network (DQN)** sobre el entorno `LunarLander-v3` de Gymnasium.

- **Fecha:** 2026-06-03
- **Entorno:** `LunarLander-v3` (8 estados continuos, 4 acciones discretas)
- **Criterio de "resuelto":** +200 de recompensa media sobre 100 episodios
- **Evaluación:** política determinista (`deterministic=True`), 20 episodios por agente

---

## 1. Configuración de los agentes

| Parámetro            | Q-Learning                      | DQN                          |
|----------------------|---------------------------------|------------------------------|
| Episodios entrenados | 10 000                          | 600                          |
| Representación       | Tabla Q discretizada (10 bins)  | MLP 8→128→128→4 (18 180 par.)|
| Learning rate        | 0.01                            | 0.001                        |
| Gamma (descuento)    | 0.999                           | 0.99                         |
| Epsilon final        | 0.6065                          | 0.0494                       |
| Estados visitados    | 6 357                           | — (aproximación continua)    |
| Experience replay    | No                              | Sí (buffer 100 000)          |
| Target network       | No                              | Sí (sync cada 10 episodios)  |
| Dispositivo          | CPU                             | CPU                          |

> Nota: con `epsilon_decay = 0.99995`, tras 10 000 episodios el agente
> Q-Learning sigue explorando bastante (ε ≈ 0.61), lo que limita su política.

---

## 2. Resumen de resultados (20 episodios de evaluación)

| Métrica                         | Q-Learning      | DQN             |
|---------------------------------|-----------------|-----------------|
| Recompensa media ± desv.        | **-289.23 ± 146.49** | **+120.74 ± 106.57** |
| Mejor episodio                  | -83.78          | +232.42         |
| Peor episodio                   | -593.89         | -128.31         |
| Aterrizajes (LANDED)            | 0 / 20 (0 %)    | 15 / 20 (75 %)  |
| Truncados (límite de tiempo)    | 0 / 20          | 4 / 20 (20 %)   |
| Estrellados (CRASHED)           | 20 / 20 (100 %) | 1 / 20 (5 %)    |
| Episodios con recompensa > 0    | 0 / 20 (0 %)    | 16 / 20 (80 %)  |

**Conclusión:** el DQN aprende una política de aterrizaje funcional
(media positiva, 75 % de aterrizajes), mientras que el Q-Learning con
discretización no logra aterrizar en ninguna corrida y siempre se estrella.

---

## 3. Detalle por episodio

### Q-Learning

| Ep | Resultado | Pasos | Recompensa |
|---:|-----------|------:|-----------:|
| 1  | CRASHED | 407 | -272.31 |
| 2  | CRASHED | 196 | -486.29 |
| 3  | CRASHED | 173 | -470.41 |
| 4  | CRASHED | 319 | -401.91 |
| 5  | CRASHED | 195 | -172.13 |
| 6  | CRASHED | 287 | -474.28 |
| 7  | CRASHED | 519 | -333.01 |
| 8  | CRASHED | 338 | -201.90 |
| 9  | CRASHED | 341 | -254.65 |
| 10 | CRASHED | 109 | -144.23 |
| 11 | CRASHED | 304 | -220.07 |
| 12 | CRASHED | 171 | -520.16 |
| 13 | CRASHED | 281 | -147.00 |
| 14 | CRASHED | 189 | -184.07 |
| 15 | CRASHED | 177 | -593.89 |
| 16 | CRASHED | 307 | -190.04 |
| 17 | CRASHED | 195 | -127.74 |
| 18 | CRASHED | 447 | -83.78 |
| 19 | CRASHED | 396 | -222.49 |
| 20 | CRASHED | 168 | -284.22 |

**Media: -289.23 ± 146.49**

### DQN

| Ep | Resultado | Pasos | Recompensa |
|---:|-----------|------:|-----------:|
| 1  | TRUNCATED | 1000 |  -22.44 |
| 2  | LANDED    |  760 | +170.62 |
| 3  | LANDED    |  883 | +100.12 |
| 4  | LANDED    |  366 | +232.42 |
| 5  | LANDED    |  635 | +189.57 |
| 6  | LANDED    |  670 |  +91.50 |
| 7  | LANDED    |  409 | +194.37 |
| 8  | LANDED    |  565 | +153.79 |
| 9  | LANDED    |  437 | +228.13 |
| 10 | TRUNCATED | 1000 |  +45.20 |
| 11 | LANDED    |  674 | +214.44 |
| 12 | LANDED    |  590 | +207.32 |
| 13 | CRASHED   |  994 | -128.31 |
| 14 | LANDED    |  476 | +163.40 |
| 15 | LANDED    |  491 | +207.04 |
| 16 | LANDED    |  420 | +200.43 |
| 17 | LANDED    |  162 |  +25.80 |
| 18 | TRUNCATED | 1000 |  -25.39 |
| 19 | TRUNCATED | 1000 |  -48.64 |
| 20 | LANDED    |  588 | +215.35 |

**Media: +120.74 ± 106.57**

---

## 4. Interpretación

- **DQN** alcanza recompensas positivas y aterrizajes consistentes. Aún no
  cruza el umbral de "resuelto" (+200 de media), pero está cerca en sus mejores
  episodios (+232, +228, +215). Con más episodios de entrenamiento debería
  estabilizarse por encima de +200.
- **Q-Learning** sufre por la **maldición de la dimensionalidad**: discretizar
  6 dimensiones continuas en 10 bins genera un espacio enorme y disperso
  (6 357 estados visitados de millones posibles), por lo que generaliza mal.
  Además, su epsilon alto (≈0.61) mantiene mucha exploración aleatoria.
- La **aproximación de función** del DQN (red neuronal) generaliza sobre el
  espacio continuo, algo que la Q-table no puede hacer; de ahí la enorme
  diferencia de desempeño (≈410 puntos de recompensa media).

---

## 5. Cómo reproducir

```bash
# Entrenar
rlgames train qlearning --episodes 10000
rlgames train dqn --episodes 600

# Simular (evaluación determinista, 20 episodios)
rlgames sim qlearning --episodes 20 --steps 0
rlgames sim dqn --episodes 20 --steps 0

# Ver info del agente / evaluación rápida
rlgames load qlearning --eval
rlgames load dqn --eval
```
