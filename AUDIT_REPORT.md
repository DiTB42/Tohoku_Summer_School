# 実装監査レポート — 論文再現性チェック

対象論文: Jiang et al., *Hierarchical RL-Guided Large-scale Navigation of a Snake Robot*, arXiv:2312.03223v1

このレポートは別セッションでの修正作業に引き継ぐためのタスクリストです。各項目に
`file:line` の参照、問題内容、根本原因、推奨修正方針を記載しています。

分類:
- **[FIX] 明確なバグ** — 必ず修正する
- **[FIX-DEV] 論文との不整合で動作不良の原因になりうるもの** — 修正する
- **[KEEP] README記載済みの簡略化** — 現状維持(触らない)
- **[CLEANUP] コード整理**

---

## [FIX] 明確なバグ(必ず修正)

### 1. CPGの振幅2階系ダイナミクスが無効化されている
- 場所: [snake_cpg.py:84-100](snake_cpg.py#L84-L100)
- 問題: 論文式(9) `r̈ = a・[a/4・(R−r) − ṙ]` を実装せず、コメントアウトして
  `x = self.R * sin(φ) + δ` と振幅を毎ステップ即座にジャンプさせている。
- 根本原因: `_r_system` メソッドは定義されているが呼び出し側で使われていない。`self.r`/`self.r_dot`
  もコメントアウトされたまま。
- 推奨修正: `self.r`, `self.r_dot` を復活させ、`update()` 内で `r_ddot = a*(a/4*(R-r) - r_dot)` を
  オイラー(またはRK4、コメントに残骸あり)積分し、`x = r * sin(phi) + delta` に戻す。
  `set_parameters()` で `R` が変わったときに `r` が滑らかに追従することを確認する。

### 2. CPGの積分dtとMuJoCo物理タイムステップが10倍ずれている
- 場所: [env_snake.py:44](env_snake.py#L44)(`dt=1/50=0.02s`)、
  [env_snake.py:183-188](env_snake.py#L183-L188)(ループ)、
  `scenes/scene.xml` / `scenes/snake.xml` に `<option timestep=...>` の指定なし(MuJoCoデフォルト0.002s)。
- 問題: ループは `cpg.update()`(仮想時間0.02s進む)→ `mj_step()` 1回(物理時間0.002sしか進まない)
  を対にしている。CPGの位相は物理より10倍速く進んでしまう。
- 推奨修正: 以下いずれかで整合させる。
  - (a) `scene.xml` に `<option timestep="0.02" .../>` を明示し、CPGのdtと一致させる
    (ただし物理安定性が落ちる可能性があるので要検証)。
  - (b) CPGのdtを `model.opt.timestep` に合わせ、`cpg.update()` を `mj_step()` と同じ頻度で
    呼ぶようにする(推奨: こちらの方が物理を粗くしないので安全)。
  - どちらを選んでも、README/Limitationsにある「maze produces NaN errors」の一因である
    可能性が高いので、修正後に複雑な迷路でのNaN再現テストを行うこと。

### 3. 報酬に計算されているのに使われていないペナルティが3つある
- 場所: [env_snake.py:213-244](env_snake.py#L213-L244)(`self_collision_penalty`, `penalty`(頭尾距離),
  `wall_penalty` を計算)、最終式 [env_snake.py:253](env_snake.py#L253)。
  ```python
  reward = (10.0*r1) + (2000.0*r2) + (0.1)*r3 + self.cumulative_waypoint_reward \
           + goal_reward + control_penalty + (100)*r_angle_alignment #+ 20)penalty #+ wall_penalty
  ```
- 問題: 3項とも計算だけして加算されていない。自己衝突・壁衝突・とぐろ巻きへの罰則が
  実質機能していない。
- 推奨修正: 3項を最終reward式に組み込む。ただし重みは (4) で作る設定可能な仕組みに載せること。

### 4. `cumulative_waypoint_reward` が到達後も毎ステップ加算され続ける
- 場所: [env_snake.py:97](env_snake.py#L97)(episodeリセット時のみ0に戻る)、
  [env_snake.py:207-208](env_snake.py#L207-L208)(到達時+300)、
  [env_snake.py:253](env_snake.py#L253)(毎ステップreward加算)。
- 問題: 一度ウェイポイントに到達すると、その後は何もしなくても+300/stepが乗り続ける。
  「到達時に一度だけ加点」の意図だったと考えられるが、実装は累積値を毎ステップ加算する形になっている。
- 推奨修正: 到達したステップでのみ `goal_reward` 同様にワンショットのボーナスを加算し、
  `cumulative_waypoint_reward` を毎ステップ足す設計をやめる
  (例: `waypoint_bonus_this_step` のようなローカル変数にして、そのステップの reward にだけ加える)。

### 5. `self.cpg` がエピソードをまたいでリセットされない
- 場所: [env_snake.py:44-45](env_snake.py#L44-L45)(`__init__`で1回だけ生成)、
  `reset()`([env_snake.py:63-107](env_snake.py#L63-L107))ではCPGの位相・状態を再初期化していない。
- 問題: 前エピソード終了時の位相 `φ`(・振幅dynamics再実装後は `r`, `r_dot` も)が次エピソードの
  先頭に持ち越される。同じ観測・行動でも出力される関節ターゲットがエピソードごとに変わり得る。
- 推奨修正: `reset()` の中で `self.cpg = PaperCPG(...)` を作り直すか、`PaperCPG` に
  `reset_state()` を追加して `phi`(と振幅系 `r`, `r_dot`)をゼロ/初期値に戻す。

### 6. ウェイポイント「到達」判定と目標地点の不整合
- 場所: `_get_current_target()` [env_snake.py:140-149](env_snake.py#L140-L149)(現在ウェイポイントと
  次ウェイポイントの**中点**を返す)、到達判定 [env_snake.py:195-211](env_snake.py#L195-L211)
  (その中点との距離が0.7未満で「到達」とみなす)。
- 問題: 実際のウェイポイント座標ではなく中点への接近で到達フラグを立てており、
  ウェイポイントの取りこぼし・過剰検知が起きうる。
- 推奨修正: 到達判定は `self.path_waypoints_world[self.current_waypoint_index]` の実座標との
  距離で行う。中点(lookahead target)は報酬の距離シェーピング(`r1`, `r2`)にのみ使う、
  という形で役割を分離する。

### 7. `test_model.py` に開発者個人の絶対パスがハードコード
- 場所: [test_model.py:7](test_model.py#L7)
  `SAC.load("/home/charlotte/Documents/SNAKE_code/TEST/models/sac_snake_final.zip")`
- 推奨修正: `SAC.load("models/sac_snake_final.zip")` など相対パスに変更。

### 8. デバッグ用printの残骸
- 場所: [env_snake.py:170](env_snake.py#L170) `print(self.current_step)`
- 推奨修正: 削除、または `verbose` フラグ配下に移動。

---

## [FIX-DEV] 論文との不整合で、動作不良を招いている可能性が高いもの

### 9. CPG位相差配列 `theta` が定数ではなく `(i+1)*θ` の増加列になっている
- 場所: [snake_cpg.py:52-57](snake_cpg.py#L52-L57) `_create_phase_shifts`
  ```python
  theta[i] = (i+1) * phase_shift
  ```
- 論文根拠: IV.D.3節「we set each parameter of the four mentioned with one single value」
  — R・ω・θ・δ はいずれもチャンネル間で単一値を複製する設計。位相差 `θ` は
  `B・θ` の形でφの隣接差分の目標値として働くため、定数でないと一様な進行波
  (serpenoid wave)にならず、関節間の位相差が際限なく増大する方向に引っ張られる。
- 推奨修正: `theta = np.full(self.n_joints - 1, phase_shift)` のように定数ベクトルへ変更。
  README実験(Exp1-3で"unstable/erratic"と報告)の一因になっている可能性が高い。

### 10. 観測空間にロボットの向き(IMU/相対姿勢)情報が一切ない
- 場所: `_get_obs()` [env_snake.py:150-166](env_snake.py#L150-L166)
  (joint_pos(12) + joint_vel(12) + head_to_target_vec(3) = 27次元)。
- 論文根拠: IV.D.2節の状態空間は joint pos(R^n) + IMU(R³) + 相対位置(R³) +
  相対姿勢 axis-angle(R⁴) = 21次元。IMUと相対姿勢が含まれる。
- 問題: 報酬には向き整合項 `r_angle_alignment = cos(angle_to_target)`
  ([env_snake.py:237](env_snake.py#L237))があるのに、観測にはロボットの向きを直接示す
  情報が存在しない(位置ベクトルと関節列から間接的に推定するしかない)。
  報酬が評価している量を方策が直接観測できないのは学習を不安定にしやすい。
- 推奨修正: `_get_snake_heading_angle()` で計算している角度(または head の回転行列/軸角表現)を
  観測に追加する。IMU相当の情報(角速度など、`data.qvel` のfree jointの回転成分などから取得可)も
  追加を検討。次元数が変わるので `observation_space` の shape も合わせて更新すること。

---

## [KEEP] README記載済みの簡略化(触らない)

- **行動空間を θ のみ(R=1, ω=1, δ=0固定)に縮小**していること
  ([env_snake.py:32](env_snake.py#L32))。README「Experiments」節に明記された意図的な実験結果。
- **DDPGではなくSACを採用**していること。README「System Overview」「Technical Details」に明記。
- **車輪付き・全関節yaw軸のみの平面ヘビ(単一CPG)**という、COBRA(pitch/yaw2CPG・sidewinding)とは
  異なるハードウェアモデルであること([scenes/snake.xml](scenes/snake.xml))。README冒頭で
  「2024年度のジェスチャー操作ベースを踏襲」と明記されており、ベースハードウェア自体が別物という
  前提。ここを論文のCOBRA仕様に合わせて作り変えるのはスコープ外。
- **迷路生成が`height=3`で実質1本道(corridor)しか作れないこと**
  ([env_snake.py:35](env_snake.py#L35)、生成ロジックは[make_maze.py:9-29](make_maze.py#L9-L29))。
  再帰バックトラッカーは2マス単位で分岐するため、複数の「セル行」を持つには高さが奇数で5以上
  必要だが、現状の`height=3`では縦方向の通路が原理的に生成されない。技術的には見落としに見えるが、
  README「Limitations」に明記された既知の制約であり、「Future Work」に「まず全迷路対応、その後
  ウェイポイントなしでの学習」という順序が明記されている。→ **現状維持でよい。将来full maze対応に
  着手する際にまとめて見直す。**

---

## [CONFIG] YAML設定化の設計方針

報酬設計・CPGハイパラ・SAC学習ハイパラなど、これまでコード中にマジックナンバーとして
埋め込まれていた値は、`config/` フォルダ配下のYAMLファイルで設定できる形式に変更する。
以下を設計方針として引き継ぐ。

### 方針
- リポジトリ直下に `config/` フォルダを新設し、デフォルト設定を `config/default.yaml` に置く。
  実験ごとに差分だけ書いた `config/exp1.yaml` のようなファイルを追加し、
  `python train_sac.py --config config/exp1.yaml` のように起動時に指定できるようにする。
- 読み込みは `PyYAML`(`requirements.txt` に既にあるか確認、無ければ追加)で行い、
  `load_config(path) -> dict` のようなヘルパーを新設(例: `config_utils.py`)。
  内部的には dataclass(`RewardConfig`, `CPGConfig`, `TrainingConfig`, `EnvConfig`)に変換して
  型安全にアクセスできるようにすると良い。
- `SnakeEnv.__init__` に `config: dict | None = None` を追加し、YAMLから読んだ値を
  デフォルト値の代わりに使う(configを渡さない場合は現状のデフォルト値にフォールバック)。
- `train_sac.py` も同様に config から SAC のハイパラを読み込む形に変更する。

### YAMLに出すべきパラメータ一覧

**reward セクション**(現状 [env_snake.py:253](env_snake.py#L253) 周辺にハードコードされている係数。
項目3で結線する自己衝突・壁衝突・頭尾距離の重みもここに含める):
```yaml
reward:
  w_progress: 10.0        # r1: 1/(0.1+距離) の重み
  w_velocity: 2000.0      # r2: 接近速度の重み
  w_smoothness: 0.1       # r3: 行動変化ペナルティの重み
  w_heading: 100.0        # 向き整合(cos(angle_to_target))の重み
  w_control: -0.01        # 制御入力の二乗ペナルティの重み
  waypoint_bonus: 300.0   # ウェイポイント到達時のワンショットボーナス(項目4修正後)
  goal_bonus: 500.0       # ゴール到達ボーナス
  w_self_collision: -5.0  # 自己衝突ペナルティ(項目3で結線)
  w_wall_collision: -75.0 # 壁衝突ペナルティ(項目3で結線)
  w_tail_proximity: -100.0 # 頭尾接近ペナルティ(項目3で結線)
  waypoint_threshold: 0.7  # ウェイポイント/ゴール到達とみなす距離
```

**cpg セクション**(現状 [env_snake.py:45](env_snake.py#L45) や `snake_cpg.py` 内のデフォルト引数):
```yaml
cpg:
  frequency_hz: 50.0       # CPG更新頻度(dt = 1/frequency_hz)
  convergence_rate: 63.0   # a パラメータ
  coupling_strength: 3.0   # mu パラメータ
  base_amplitude: 1.0      # R(固定運用時)
  base_omega: 1.0          # ω(固定運用時)
  base_delta: 0.0          # δ(固定運用時)
  base_phase_shift: 0.7854 # θ の初期/固定値(π/4)。項目9修正後は定数ベクトルに展開される元の値
```

**training セクション**(現状 [train_sac.py:32-56](train_sac.py#L32-L56) にハードコード):
```yaml
training:
  algorithm: SAC
  learning_rate: 1.0e-3
  batch_size: 1028
  buffer_size: 9_000_000
  tau: 0.18
  gamma: 0.5
  ent_coef: auto
  train_freq: [1, step]
  gradient_steps: 1
  total_timesteps: 10000
  log_interval: 4
  tensorboard_log: sac_snake_tensorboard/
  model_save_path: models/sac_snake_final
```
  ※ `gamma=0.5`, `tau=0.18` は現状かなり特異な値(標準的には gamma≈0.99, tau≈0.005)。
  YAML化した上で、まずは標準的な値との比較実験をしやすくすることが目的。

**env セクション**(現状 [env_snake.py:18-49](env_snake.py#L18-L49) にハードコード):
```yaml
env:
  cpg_frequency: 50
  rl_frequency: 0.1
  max_episode_steps: 300
  maze_height: 3           # KEEP扱い(corridor限定)。将来full maze対応時にここを変える想定
  maze_width: 7
  action_low: 0.5          # θ の下限(README記載の縮小行動空間)
  action_high: 1.0         # θ の上限
```

### 実装時の注意
- 既存のデフォルト値は変えず、「今と同じ挙動をYAML経由で再現できる」ことをまず確認してから
  値の調整実験に入ること。
- 項目3・4(報酬の死んだペナルティ・累積バグ)の修正と、このYAML化は同時に行う
  (係数を結線する際に、ハードコードではなくconfigから読むように書く)。
- `RewardConfig` 等のdataclassにデフォルト値を持たせておけば、YAMLファイルが指定されない場合や
  一部キーが欠けている場合でも安全に動作する。

---

## [CLEANUP] コード整理

### 13. 未使用の `stable_baselines3` ソース丸ごとコピーが残っている
- 場所: ルート [`__init__.py`](__init__.py)、[`sac/`](sac/) ディレクトリ一式
  (`sac/sac.py`, `sac/policies.py`, `sac/__init__.py`)。
- 問題: `train_sac.py`・`test_model.py` はどちらも `from stable_baselines3 import SAC` を使っており、
  これらのローカルコピーはどこからも参照されていない(確認済み: `from sac import` / `import sac` は
  リポジトリ内に一切なし)。しかもルート `__init__.py` は存在しない `version.txt` を読もうとするため、
  誤って実行するとエラーになる。
- 推奨修正: 両方削除する。将来SACをカスタマイズする予定がある場合のみ、明示的にその意図を
  README等に記載した上で残す。

### 14. READMEに書かれたディレクトリ構成と実際のファイル配置が食い違っている
- 場所: README「Repository Structure」節([README.md:51-68](README.md#L51-L68))は
  `mazes/make_maze.py`, `mazes/mujoco_tools.py`, `cpg/snake_cpg.py` のようなサブディレクトリ構成を
  示しているが、実際は全てリポジトリ直下にフラットに置かれている。
- 推奨修正: どちらかに合わせる(ファイルをサブディレクトリに移動するか、READMEの構成図を実態に
  修正する)。import文の変更が発生するため、移動する場合は影響範囲(`env_snake.py` の
  `from make_maze import ...` など)を確認すること。

---

## 修正時の推奨順序

1. 項目2(dt不整合)→ 項目5(CPGリセット)→ 項目1(振幅ダイナミクス復活)の順でCPG/物理まわりを
   直してから、実際に迷路でNaNが再現するか確認する。
2. [CONFIG]のYAML化(`config/`フォルダ・dataclass・ローダー実装)を先に用意し、
   項目3・4(報酬の死んだペナルティ・累積バグ)の修正をそのconfig経由で行う。
3. 項目6(到達判定)、項目9(位相差配列)、項目10(観測空間拡張)を修正し、短時間の学習で
   挙動が改善するか(README Exp1-3相当の再実験)を確認する。
4. 項目7・8・13・14は影響範囲が小さいので任意のタイミングでまとめて片付けてよい。
   迷路の`height`は現状維持([KEEP]参照)。
