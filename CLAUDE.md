# CLAUDE.md — 開発ルール

## Approach
- Think before acting. Read existing files before writing code.
- Be concise in output but thorough in reasoning.
- Prefer editing over rewriting whole files.
- Do not re-read files you have already read unless the file may have changed.
- Test your code before declaring done.
- No sycophantic openers or closing fluff.
- Keep solutions simple and direct.
- User instructions always override this file.

---

## 🔴 絶対禁止（確認なしに実行してはいけないこと）

以下は「ユーザーが指示した」場合でも、必ず一度立ち止まって
「本当にこれを実行していいですか？影響範囲：○○」と確認してから行うこと。

```
# ファイル削除
rm -rf
unlink

# DB破壊
DROP TABLE
TRUNCATE
DELETE FROM（WHERE句なし）
UPDATE（WHERE句なし）

# Git上書き
git push --force
git push -f
→ 代わりに git push --force-with-lease を使う

# 本番DB・本番環境への直接操作
```

> **設定補足（settings.json と併用推奨）**
> このファイルだけでは物理的に止められない。
> プロジェクトルートの `.claude/settings.json` に以下を設定すること：
> ```json
> {
>   "permissions": {
>     "deny": [
>       "Bash(rm -rf*)",
>       "Bash(git push --force *)",
>       "Bash(git push -f *)"
>     ]
>   }
> }
> ```

---

## 🏗️ 環境判定ルール（本番 vs 開発）

作業開始時に必ず環境を確認すること。

```
APP_ENV=production  → 本番環境（破壊的操作前に確認必須）
APP_ENV=development → 開発環境（自由に操作可）
APP_ENV=staging     → ステージング環境（本番に近い検証用）
```

**本番環境で破壊的操作を行う場合は必ず確認を挟む：**

```python
# Python の場合
import os, sys
if os.environ.get("APP_ENV") == "production":
    answer = input("⚠️ 本番環境で操作します。続けますか？ (yes): ")
    if answer != "yes":
        print("中止しました")
        sys.exit(0)
```

```javascript
// Node.js の場合
if (process.env.APP_ENV === "production") {
    const answer = await question("⚠️ 本番環境で操作します。続けますか？ (yes): ");
    if (answer !== "yes") process.exit(0);
}
```

絶対にやってはいけないこと：本番環境で「ちょっと試す」「動作確認する」

---

## 📦 ライブラリのインストールルール

新しいライブラリを追加・更新する前に、必ず以下の手順を実行すること。
「有名なライブラリだから安全」「以前使ったことがあるから大丈夫」は理由にならない。
正規の人気パッケージが突然汚染されたケース（litellm、telnyx 等）が実際に起きている。

### ステップ1：web検索で安全性を確認する

以下のキーワードで検索し、問題がないか確認する：
- `{ライブラリ名} security`
- `{ライブラリ名} malware`
- `{ライブラリ名} compromised`

確認する観点：
- リリース履歴に不審な点がないか（GitHubタグなしでパッケージレジストリだけ更新されていないか）
- セキュリティDBやニュースで最近言及されていないか
- 作者アカウントが新規・パッケージ数1つだけ・GitHub情報なしでないか
- 最終更新が長期間止まっていないか

### ステップ2：確認結果をユーザーに報告する

インストールを進める前に、必ず以下の形式で報告する：

問題なし（進めてよい場合）：
「{ライブラリ名} {バージョン} を確認しました。既知の問題なし、GitHubタグ対応済みです。インストールを進めます。」

問題あり（進めてはいけない場合）：
「{ライブラリ名} {バージョン} に問題を確認しました：{理由}。代替として {代替ライブラリ} を提案します。」

### ステップ3：バージョンをピン留めしてインストールする

```bash
# Python — 良い例
pip install requests==2.31.0

# Node.js — 良い例
npm install express@4.18.2

# バージョン未指定は禁止（最新版が自動で入る）
pip install requests
npm install express
```

`requirements.txt` / `package.json` / `pyproject.toml` にも必ずバージョンを明記する。

### ステップ4：推移依存（間接依存）も確認する

```bash
# Python
pip install pipdeptree && pipdeptree

# Node.js
npm list
```

### 適用しないケース

- 言語標準ライブラリ（Python の os・sys・json、Node.js の fs・path 等）
- すでにロックファイルに固定されているライブラリ（バージョン変更がない場合）

---

## 🔐 シークレット・APIキー管理

- APIキー・トークン・パスワードはコードに直書き禁止
- 環境変数（`.env`）から読み込む
- プロンプト内にAPIキーやトークンを貼り付けない（ログファイルに残る）
- エラーログ・デバッグ出力にAPIキーや個人情報を含めない

```python
# Python — 正しい読み込み方
import os
API_KEY = os.environ.get("API_KEY")
if not API_KEY:
    raise RuntimeError("API_KEY が設定されていません")
```

```javascript
// Node.js — 正しい読み込み方
const apiKey = process.env.API_KEY;
if (!apiKey) throw new Error("API_KEY が設定されていません");
```

**必ず `.gitignore` に含めるもの：**
```
.env
.env.*
*.local
# DB
*.db
*.sqlite
*.sqlite3
# 設定ファイル（認証情報を含む可能性があるもの）
*-config.json
*-credentials.json
secrets/
# バックアップ
*.backup_*
```

**外部サービスのAPIキー制限：**
- 各APIキーは使用するAPIのみに制限する（「制限なし」は禁止）
- キーは用途ごとに分離する（Firebase用・地図API用・AI API用 等）
- 特にAI系API（Gemini・OpenAI等）は**サーバーサイドでのみ使用**する
  （クライアントサイドに含めると課金爆発の原因になる）

---

## 🗄️ データベース

### クエリ記述ルール

```python
# Python（SQLite / PostgreSQL）— 正しい書き方
cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))

# ❌ 危険：文字列結合（SQLインジェクション）
cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")
```

```javascript
// Node.js（PostgreSQL）— 正しい書き方
await db.query("SELECT * FROM users WHERE id = $1", [userId]);

// ❌ 危険
await db.query(`SELECT * FROM users WHERE id = ${userId}`);
```

- `DELETE` / `UPDATE` には必ず `WHERE` 句をつける
- `WHERE` 句なしの `DELETE` / `UPDATE` は絶対に書かない

### 本番DB操作前の必須手順

以下の操作を行う**前に必ずバックアップを取る**：
`ALTER TABLE` / `DROP` / `UPDATE` / `DELETE` / `INSERT`（一括）/ マイグレーション

```bash
# SQLite
cp app.db app.backup_$(date +%Y%m%d_%H%M%S).db

# PostgreSQL
pg_dump $DATABASE_URL > backup_$(date +%Y%m%d_%H%M%S).sql

# バックアップを確認してから操作
ls -lh *.backup_* *.sql
```

- スキーマ変更はステージング環境で先に試す
- マイグレーションは冪等にする（何度実行しても同じ結果になること）
- リストア手順を事前に確認しておく（試したことのない復旧手順は存在しないのと同じ）

---

## 🛡️ セキュリティ実装ルール

### IDOR防止（必ず実装）

データを取得・更新・削除するたびに、現在のユーザーのものかサーバー側で確認する。

```python
# Python（Flask）— 正しい書き方
@app.route("/item/<int:item_id>/edit", methods=["POST"])
def edit_item(item_id):
    item = db.get_item(item_id)
    if item is None:
        abort(404)
    if item["user_id"] != session["user_id"]:  # ← 必ずこのチェック
        abort(403)
    # 以降、編集処理

# ❌ 危険：IDをそのまま使って更新
def edit_item(item_id):
    db.update_item(item_id, request.form)  # 誰でも他人のデータを書き換えられる
```

```javascript
// Node.js（Express）— 正しい書き方
app.put("/item/:id", async (req, res) => {
    const item = await db.getItem(req.params.id);
    if (!item) return res.status(404).json({ error: "Not found" });
    if (item.userId !== req.session.userId) {  // ← 必ずこのチェック
        return res.status(403).json({ error: "Forbidden" });
    }
    // 以降、更新処理
});
```

### 入力バリデーション

クライアント側のバリデーションはUXのためだけ。**セキュリティとしては無意味。**
サーバー側で必ず同じ検証を行う。

```python
# Python — サーバー側バリデーション例
text = request.form.get("comment", "").strip()
if not text:
    abort(400, "入力が空です")
if len(text) > 500:
    abort(400, "500文字以内で入力してください")
```

```javascript
// Node.js — サーバー側バリデーション例
const text = req.body.comment?.trim() ?? "";
if (!text) return res.status(400).json({ error: "入力が空です" });
if (text.length > 500) return res.status(400).json({ error: "500文字以内で入力してください" });
```

- ファイルアップロードは拡張子・MIMEタイプ・サイズをサーバー側で検証する
- URLパラメータは整数であることを確認してからDBクエリに使う

### エラーハンドリング

```python
# Python（Flask）— 本番向け
@app.errorhandler(500)
def internal_error(e):
    app.logger.error(f"Internal error: {e}", exc_info=True)  # ログには詳細を記録
    return jsonify({"error": "処理に失敗しました"}), 500    # ユーザーには抽象的なメッセージ
```

```javascript
// Node.js（Express）— 本番向け
app.use((err, req, res, next) => {
    console.error(err);                                      // ログには詳細を記録
    res.status(500).json({ error: "処理に失敗しました" });  // ユーザーには抽象的なメッセージ
});
```

本番では必ず `DEBUG=False` / `NODE_ENV=production` を維持する。

### セキュリティヘッダー

以下のヘッダーを設定・維持する：
```
X-Frame-Options: DENY
X-Content-Type-Options: nosniff
Strict-Transport-Security: max-age=31536000; includeSubDomains
Content-Security-Policy: （プロジェクトに合わせて設定）
```

### その他

- セッションCookieは `HttpOnly` / `SameSite=Strict` を設定する
- すべてのフォーム・APIにCSRF保護を実装する
- ログイン・登録・投稿等の重要APIにはレート制限を設ける
- 新しいAPIエンドポイントを追加する際も同様にレート制限を設ける

---

## 📝 ロギングルール

```python
# ✅ 安全なログ
logger.info(f"アイテム登録: user_id={user_id}, item_id={item_id}")
logger.error(f"DB接続エラー: {type(e).__name__}")

# ❌ 危険：個人情報・シークレットをログに出力
logger.info(f"ログイン: email={email}, password={password}")
logger.debug(f"API呼び出し: key={api_key}")
```

ログに含めてはいけないもの：
- パスワード・APIキー・トークン・セッションID
- メールアドレス・氏名・住所・電話番号などの個人情報
- クレジットカード番号・金融情報

---

## 💰 外部API・コスト管理

新しい外部API呼び出しを実装する際は、必ずリトライ上限を設ける。

```python
# Python — リトライのひな形
import time

def call_with_retry(fn, max_attempts=3):
    for i in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as e:
            if i == max_attempts:
                raise
            wait = 2 ** (i - 1)  # 1秒 → 2秒 → 4秒
            print(f"リトライ {i}/{max_attempts}: {e}. {wait}秒待機")
            time.sleep(wait)
```

```typescript
// TypeScript — リトライのひな形
async function withRetry<T>(fn: () => Promise<T>, maxAttempts = 3): Promise<T> {
    for (let i = 1; i <= maxAttempts; i++) {
        try {
            return await fn();
        } catch (err) {
            if (i === maxAttempts) throw err;
            const wait = 1000 * Math.pow(2, i - 1);
            await new Promise(r => setTimeout(r, wait));
        }
    }
    throw new Error("unreachable");
}
```

- リトライ上限なしは禁止（無限ループ・課金爆発の原因）
- AI系API（LLM・画像生成等）は特に上限設定に注意する
- 新しい外部APIを追加する前に料金体系と上限設定を確認する

---

## 🗂️ Git ルール

- `git push --force` は禁止。代わりに `git push --force-with-lease` を使う
- コミット前に `git diff --staged` でステージ内容を確認する
- シークレットファイルが含まれていないか確認する：
  ```bash
  git diff --cached --name-only | grep -E '\.(env|db|sqlite|json)$'
  ```
- GitHubリポジトリは最初は必ずプライベートにする
- コミットメッセージには「何を・なぜ」変えたかを書く

---

## ✅ 新機能追加時のチェックリスト

コードを書き終えたら、この順番で確認する：

**セキュリティ**
- [ ] ログイン済みユーザーのみアクセス可能か
- [ ] 他ユーザーのデータにアクセスできないか（IDOR確認）
- [ ] 入力値のサーバー側バリデーションはあるか
- [ ] CSRFトークンは検証されているか
- [ ] レート制限はあるか
- [ ] エラー時に内部情報（スタックトレース・SQL）を返していないか
- [ ] ログに個人情報・APIキーを含んでいないか

**シークレット・依存**
- [ ] APIキー・シークレットがコードに含まれていないか
- [ ] `.gitignore` で除外すべきファイルが除外されているか
- [ ] 新しいライブラリを追加した場合、安全性確認とバージョン固定をしたか

**コスト・安定性**
- [ ] 外部API呼び出しにリトライ上限はあるか
- [ ] 途中で処理が止まったとき、データが中途半端な状態にならないか（トランザクション）
- [ ] 同じ処理を2回実行しても問題ないか（冪等性）
- [ ] 本番DBを操作する前にバックアップを取ったか