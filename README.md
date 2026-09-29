# local realtime voice translator
これは [audio.cpp](https://github.com/0xShug0/audio.cpp) と [llama.cpp](https://github.com/ggml-org/llama.cpp) によるローカル音声認識音声合成とローカル LLM を利用して、発話者の音声をそのまま参照音声として使うことで発話者の声質そのままで日本語で話した内容を英語音声に翻訳したり、英語で話した内容を日本語音声に翻訳するなど、２言語間でリアルタイムに翻訳するアプリケーションの概念実証プログラム(PoC)です。PySide6 で UI を実装してます。Gemini Flash 3.6 と GPT6-Astra lowで作成しています。

## Windowsの場合
### インストール
セットアップやモデルダウンロード用のバッチファイルを用意してあるのでこれを利用してください。

[リポジトリのZIPファイル](https://github.com/asfdrwe/local_realtime_voice_translator/archive/refs/heads/main.zip)を押してダウンロードしてください。

適当な場所に移動させて右クリックで展開してください。展開したフォルダを開いてください。

次の順に実行してください。
- install_git_uv.bat
[uv](https://docs.astral.sh/uv/) が必要です。uv がインストールされていない場合は右クリックで管理者モードでこれを実行して git と uv をインストールするか、手動で uv をインストールしてください。git は不要ですがおまけでインストールします。
- setup.bat
ダブルクリックして実行してください。local realtime voice translator に必要な python とモジュールを uv でインストールし、音声認識・音声合成サーバーの audio.cpp の本体と framework のダウンロードとローカル LLM 実行エンジンの llama.cpp のダウンロードを行います。
- modeldownload.bat
ダブルクリックして実行してください。使用する音声認識モデル Qwen3-ASR-0.6B と音声合成モデル Qwen3-TTS-0.6B-Base と LLM gemma-4-E4B-it をダウンロードします。

### 使い方
run-tab.bat をダブルクリックしてください。LLM を動作させるタブと音声認識・音声合成を動作させるタブとこのプログラムを動作させるタブの３つのタブを持つターミナルと設定画面が表示されるはずです。設定画面はそのまま出しっぱなしても問題ないです。

日本語で話すと、発話検出して音声認識し、英語に翻訳して、英語で音声合成されます。

マイク入力とスピーカ出力の設定を行い発話最低音量やマイクゲインを適切に設定して正しく音声認識するように設定すれば、マイクに話すだけでリアルタイムに音声に翻訳されるはずです。

#### 設定項目
- サーバー設定
  - audio.cpp アドレス    audiocpp_server のアドレスやポート番号を指定してください。run-tab.batから起動している場合は変更不要です。
  - llama.cpp アドレス    llamacpp-server のアドレスとポート番号を指定してください。run-tab.batから起動している場合は変更不要です。
- オーディオデバイ
  - マイク(入力)     使用するマイクデバイスを選択してください。
  - スピーカー(出力)     使用するスピーカーやイヤホンを選択してください。
  - マイク: ON/OFF     一時的にマイクオフすることができます。
  - TTS再生中のVADソフトミュート     スピーカーで音声を再生するとマイクに入ってしまってループしてしまうのでそれを防ぐ設定です。イヤホン使用時などマイクに音声合成結果が入らないならチェックを外しても大丈夫です。
- 参照音声
  - 話者音声の動的参照を使用 & 動的参照最小長(秒)     チェックすると次の最小長以上発話している場合に参照音声として使うことで音声合成される音声に話者の声質を反映させます。
  - デフォルト参照パス     参照音声のパスです。これが正しくないないと短い発話時に参照音声がないので音声合成に失敗します。参照ボタンを押してvoice_dirフォルダのchara1.wavを選択しないとファイル認識してくれないかもしれません。
  - デフォルト参照テキスト    参照音声で発話している文章を書いてください。Qwen3 TTS base の仕様により参照音声と参照テキストの両方がないと音声合成されません。
- 発話検出(VAD) ASR設定
  - 発話最低音量(VAD)     ここが大きいと大きな声を出さないと発話として扱われれず、ここが小さいと雑音を拾いやすくなります。ターミナルにマイクの音量と発話認識するしきい値を１秒ごとに表示するようになっているので、これを見て適切に設定してください。
  - マイクゲイン(倍率)    マイクの入力が弱い場合には倍率を増やして増幅させてください
  - 無音終了時間(秒)    途中でこの秒数以上発話を止めるとそこで区切られます。なるべく区切られたくない場合は長くしてください。
  - 最短発話長(秒)    発話が短すぎる場合はノイズとして無視します。この秒数以上発話しないと音声認識しません
  - 最長発話長(秒)    発話が長すぎると翻訳や音声合成に時間がかかりメモリの消費も多くなるので。これ以上の発話を一旦区切ります。
  - 最低 ASR 信頼度    音声認識の精度を表す ASR 信頼度が低い場合に翻訳せず無視します。この信頼度の設定です。
- 言語 LLM 翻訳
  - 言語A & 言語B    互いに翻訳する対になる 2 つの言語を指定してください。
  - LLM Temperature    通常は変更する必要はないです。増やすと LLM の動作がよりランダムになります。
  - システムプロンプト    通常は変更する必要はないです。LLM への翻訳指示プロンプトです。プロンプトを初期値に戻すボタンで初期システムプロンプトに戻ります。
- 処理ログ ステータス 発話検出や音声認識内容、翻訳結果、音声合成処理結果など様々な処理内容が表示されます。エラーが表示されたらなにかおかしいので対応してください。
  - 認識結果をバルーン表示     音声認識内容がバルーンでシステムトレイ上に表示されます。多分邪魔なので OFF にしてください。
  - 翻訳結果をバルーン表示     翻訳結果がバルーンでシステムトレイ上に表示されます。多分邪魔なので OFF にしてください。　

一番下の設定を保存して適用ボタンを押さないと反映されないので、設定変更後にボタンを押してください。

### メニュー
右下のシステムトレイの VT アイコンを右クリックすると設定画面や一時的なマイクのオンオフや終了メニューがあります。

### 終了方法
設定ウィンドウを閉じてもプログラムは終了しませんので、システムトレイの VT アイコンを右クリックして終了を押してください。run-tab.bat から起動している場合はターミナルのウィンドウを閉じれば全部終了します。

## Linux や macOS の場合
### インストール
git で取得し、uv で必要な環境とモジュールのインストールをしてください。

```
git clone https://github.com/asfdrwe/local_realtime_voice_translator
cd local_realtime_voice_translator
uv venv -p 3.13
uv pip install -r requirements.txt
```

### 各種サーバーの起動
他のモデルでも動作するかもしれませんが、audio.cpp 内蔵の silero_vad で発話検出し、Qwen3 asr　で音声認識し、Qwen3 tts baseで音声合成し、gemma-4-E4B-it-Q4_K_M で翻訳することを前提としています。audio.cpp 用のモデルは [こちらから](https://huggingface.co/audio-cpp/audio.cpp-gguf)、llama.cpp用は[gemma-4-E4B-it-Q4_K_M](https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/blob/main/gemma-4-E4B-it-Q4_K_M.gguf)からダウンロードできますし、audio.cpp/tools/model_manager_v2.py を使ってダウンロードすることもできます。

audio.cpp の audiocpp_server をこれらのモデルで動かせるように起動してください。[voicetranslator.json](voicetranslator.json)は models フォルダ以下に qwen3-asr-0.6b-q8_0.gguf と qwen3-tts-12hz-0.7b-base-q8_0.gguf をダウンロードし、cuda バックエンドでポート番号 8088 で起動するようにした設定ファイルです。audio.cpp の audiocpp_server が audio.cpp/build/bin/audiocpp_server にあり、audio.cpp 内蔵の silero_vad が audio.cpp/assets/framework/models/silero_vad/
silero_vad_16k.safetensorsではなく models/silero_vad/silero_vad_16k.safetensors にあることを前提にした設定ファイルです。この設定ファイルを使うならば、

```
audio.cpp/build/bin/audiocpp_server --config voicetranslator.json
```

です。

llama.cpp の llama-server を gemma-4-E4B-it-Q4_K_M モデルを使うように起動させてください。models フォルダに gemma-4-E4B-it-Q4_K_M.gguf があり、llama.cpp の llama-server のパスが llama.cpp/build/bin/llama-server ならば

```
llama.cpp/build/bin/llama-server -m models/gemma-4-E4B-it-Q4_K_M.gguf -c 4096
```

です。llama-server が VRAM をたくさん確保して audiocpp-server が使う VRAM がなくなってしまう場合があるので、-c 4096 をつけてコンテキスト長を下げるなど llama-server の VRAM 消費を抑える設定にしてください。

### 起動
uv でセットアップしたならば、

```
uv run python main.py
```

### 使い方
上記のWindows の設定方法をみてください。macOS では認識内容や翻訳内容のバルーン表示はされないと思います。

### メニュー
システムトレイ(Linux の KDE なら右下、macOSなら右上)の VT アイコンを右クリックすると設定画面やマイクのオンオフや終了メニューがあります。

### 終了方法
設定ウィンドウを閉じてもプログラムは終了しませんので、システムトレイの VT アイコンを右クリックして終了を押してください。

## ライセンス / License
このリポジトリの成果物は [CC0 1.0 全世界 (CC0 1.0) パブリック・ドメイン提供](https://creativecommons.org) のもとで公開します。

著作権法上のすべての権利を放棄しているため、商用利用、改変、再配布を含め、いかなる目的でも許可なく自由に使用できます。

This project is licensed under the [CC0 1.0 Universal (CC0 1.0) Public Domain Dedication](https://creativecommons.org).
