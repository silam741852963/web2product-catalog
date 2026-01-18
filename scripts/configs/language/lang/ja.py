from __future__ import annotations

LANG_CODE = "ja"

# Additions only. English base always remains active.
LANG_SPEC = {
    "PRODUCT_TOKENS": [
        "製品",
        "サービス",
        "ソリューション",
        "プラットフォーム",
        "機能",
        "仕様",
        "カタログ",
        "型番",
        "モデル",
        "データシート",
        "見積",
        "見積もり",
        "価格",
        "購入",
        "注文",
        "導入",
    ],
    "EXCLUDE_TOKENS": [
        "ニュース",
        "お知らせ",
        "ブログ",
        "採用",
        "会社概要",
        "IR",
        "投資家",
        "プレス",
        "イベント",
        "サポート",
        "お問い合わせ",
        "利用規約",
        "プライバシー",
        "クッキー",
    ],
    "INTERSTITIAL_PATTERNS": [
        r"\bアクセス\s*拒否\b",
        r"\b権限がありません\b",
        r"\b認証が必要\b",
        r"\bログイン\b",
        r"\b利用規約\b",
        r"\bプライバシーポリシー\b",
        r"\bCookie\b",
        r"\bクッキー\b",
        r"\b人間であることを確認\b",
        r"\bCAPTCHA\b",
    ],
    "COOKIEISH_PATTERNS": [
        r"\bクッキー\b",
        r"\bCookie\b",
        r"\b同意\b",
        r"\bプライバシー\b",
        r"\b設定\b",
    ],
    # Optional: only if you want Japanese stopwords for scoring
    "STOPWORDS": {
        "株式会社",
        "有限会社",
        "公式",
        "サイト",
        "ページ",
    },
}
