# 実習課題：論文に実装を近づけよう

このリポジトリは論文
[*Hierarchical RL-Guided Large-scale Navigation of a Snake Robot*](https://arxiv.org/abs/2312.03223v1)
（実習用の解説は [`COBRA_paper_JP.md`](COBRA_paper_JP.md)）を再現実装したものです。
ただし**わざと論文と一致していない部分を残して**あります。あなたの課題は、その差分を自分で実装して
論文の設計に近づけることです。

- リポジトリと論文の差分の全体像は [`COBRA_implementation_diff_JP.md`](COBRA_implementation_diff_JP.md) にまとまっています。
- コード中の該当箇所には `# TODO(学生課題1 …)` / `# TODO(学生課題2 …)` コメントが置いてあります。
- **現状のコードはそのままでも動きます**（スカラー θ のみの縮小版）。まず動かしてから拡張してください。

想定所要時間は全体で **約10時間**。課題は順番に取り組むのがおすすめです。

---

## 事前準備：環境構築と動作確認

README の手順で uv 環境を作ります。

```bash
uv venv --python 3.11
uv pip install -r requirements.txt
```

まず現状（縮小版）が動くことを確認しましょう。

```bash
# CPG 単体（RLなし）で蛇が前進するか（cpg/snake_cpg.py の __main__）
uv run python cpg/snake_cpg.py

# 学習を短時間だけ回す（config を編集して total_timesteps を小さくしてもよい）
uv run python train_sac.py
```

学習の可視化：

```bash
uv run tensorboard --logdir sac_snake_tensorboard/
```

---

## 課題1：行動空間を R⁴ に拡張する（論文 §3 / 差分項目6）

### 目標

現状、RL が出力する行動は **位相差 θ のスカラー1つだけ**で、振幅 R・周波数 ω・オフセット δ は
固定値になっています。これを論文の行動空間 **(R, ω, θ, δ) の4次元**に拡張します。

特に重要なのは **δ（オフセット）** です。δ で正弦波の中心をずらすと体全体が弓なりに曲がり、
蛇が **旋回**できるようになります（[`COBRA_paper_JP.md`](COBRA_paper_JP.md) §2.1）。今は δ=0 固定なので
実質まっすぐ進む挙動しかできません。

### 該当ファイル・箇所

| 場所 | やること |
|---|---|
| [`env_snake.py`](../env_snake.py) `__init__` の `action_space` | `Box` を1次元→4次元に。各成分の low/high を各パラメータ範囲へ |
| [`env_snake.py`](../env_snake.py) `step()` の `theta = action` | action を `R, omega, theta, delta` に分解し `cpg.set_parameters(...)` へ渡す |
| [`config/default.yaml`](../config/default.yaml) `env` の `action_low/high` | パラメータごとの範囲を持てるよう設計し直す（キー追加も可） |

`cpg.set_parameters(R, omega, theta, delta)` は既に4引数を受け取れる設計になっているので、
CPG 本体（[`cpg/snake_cpg.py`](../cpg/snake_cpg.py)）は基本的に変更不要です。

### ヒント

- 行動範囲の目安（論文値）：`R∈[0, 1.5]`, `ω∈[-0.1, 0.1]`, `θ∈[-π, π]`, `δ∈[-0.1, 0.1]`。
- **Stable-Baselines3 の SAC は tanh 出力を `Box` の low/high へ自動でスケーリング**します。
  明示的なスケーリング処理を自分で書く必要はありません（`action_space` の範囲を正しく設定すればOK）。
- いきなり RL で確認せず、まず RL を外して手でパラメータを与え、**δ を振ると旋回するか**を
  [`cpg/snake_cpg.py`](../cpg/snake_cpg.py) の `__main__` で先に確認するとデバッグが楽です
  （[`COBRA_paper_JP.md`](COBRA_paper_JP.md) §6 マイルストーン2）。
- 行動が4次元になると `last_action` との差分 `r3`（報酬）も自動的に4次元ベクトルのノルムになります。

### 確認方法（受け入れ基準）

- 手動で δ を正に振ると頭が弧を描き、進行方向が変わる。
- `uv run python train_sac.py` が4次元行動空間でエラーなく学習を回せる。

---

## 課題2：報酬の重みを設計する（論文 §4.2 / 差分項目9）

### 目標

報酬関数は論文の式(12)の**3項だけ**に整理してあります
（[`env_snake.py`](../env_snake.py) `step()`）：

```
reward = w_progress * r1  +  w_velocity * r2  -  w_smoothness * r3
```

- `r1 = 1/(0.1 + d_t)` … ゴールに近いほど大きい（近接報酬）
- `r2 = d_{t-1} - d_t`  … ゴールへ近づく速度（接近報酬）
- `r3 = ‖a_t - a_{t-1}‖` … 行動の急変ペナルティ（**減算**）

論文はこの3項の**重み係数を明記していません**。`w_progress / w_velocity / w_smoothness` を
自分で設計・調整するのがこの課題です。

### 該当ファイル・箇所

| 場所 | やること |
|---|---|
| [`config/default.yaml`](../config/default.yaml) `reward` | 3つの重みを調整（別 YAML を作って `--config` で渡してもよい） |

### ヒント

- `r1` は近傍で、`r2` は遠方で効く**相補関係**です（[`COBRA_paper_JP.md`](COBRA_paper_JP.md) §4.2）。
  両者のスケールが極端に違うと片方しか効きません。
- `r3` の重みが大きすぎると「動かないのが最適」になり蛇が止まります。小さすぎると挙動がガクガクします。
- `r1` と `r2` の距離スケール（メートル）が違う点に注意。`r2` は1ステップの変位なので値が小さめです。
- 別 config で実験する例：`uv run python train_sac.py --config config/exp1.yaml`

### 確認方法（受け入れ基準）

- TensorBoard でエピソード累積報酬やゴール到達が改善する重みの組を見つけられる。
- 蛇が「止まる／暴れる」だけでなく、ウェイポイント方向へ進む挙動が出る。

---

## 発展（余力があれば）

差分ドキュメント [`COBRA_implementation_diff_JP.md`](COBRA_implementation_diff_JP.md) には、
上記以外にも論文と食い違う点（観測の ego-centric 化、位相の mod 正規化、周波数比 1:100 など）が
整理されています。時間が余ったらそちらにも挑戦してみてください。
