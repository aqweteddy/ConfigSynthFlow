DEFAULT_EDIT_QA_TEMPLATE = {
    "edit": """### 改善對話

你將會獲得一段 **user** 和 **assistant** 的聊天紀錄，目的是基於建議，針對對話進行最佳化，確保符合高品質標準。

---

### 改善目標

根據以下建議，請對對話進行改進，請注意:
- 你必須要以第一人稱提問，並且只能有一個問號。
當你在回答時，若發現會使用到文章外的資訊，輸出無法回答即可。

{{ suggestion }}

---

### 文章

{{ text }}

### 關鍵實體
{{metatags.entities}}

### 關鍵字
{{metatags.keywords}}

### 適合任務
{{metatags.tasks}}

---


### 原始對話紀錄

{{ mes_str }}

---

### 輸出格式

務必遵守以下 JSON 格式，提供你的改進結果：

{
    "query": "改進後的問題",
    "response": "改進後的答案"
}""",
    "evolve": """### 改善對話""",
}

DEFAULT_VALID_QA_TEMPLATE = {
    "validate": """### 任務說明

你將獲得一組 **user** 和 **assistant** 的聊天紀錄以及相關的背景文章。你的任務是根據文章內容，針對對話進行評估，並基於以下標準對對話品質進行分析與打分：
---

### 評估標準
1. **問題品質（Question Quality）：**
   - 問題是否清晰、具體，並聚焦於以下主題之一：
     - 產品推薦或服務比較
     - 數據/表格的提取與分析
     - 查詢特定產品或編號細節
     - 合規性檢查
   - 問題足夠口語化的同時，是否避免模糊或過於寬泛的提問。問題應包含關鍵字或具體實體（如產品名稱、數字、編號）。
   - 是否避免了暗示文章存在的表述（如「根據文章」等）。
   - 是否避免了可能因為時間改變而使問題不再適用的內容。
   - 是否只包含一個問題，避免多重問題。

2. **答案品質（Answer Quality）：**
   - 回答是否簡潔明確，使用繁體中文且直擊問題重點，避免冗長敘述。
   - 回答是否避免提及「根據文章」等提示背景文章存在的字眼。

3. **一致性（Consistency）：**
   - 問題與答案是否相符，且答案能有效回應問題的核心需求。
   - 答案內容是否符合背景文章的事實與細節。

---

### 輸出格式

請用以下格式提供你的評估結果：

```json
{
    "reason": "簡要分析對話品質與評分理由",
    "suggestion": "對問題或答案改善的具體建議",
    "score": 整數（範圍：0-5，5 為最佳）"
}
```

---

### 聊天紀錄

{{ mes_str }}
---

### 文章

{{ text }}"""
}

DEFAULT_SELF_INSTRUCT_FROM_DOC_TEMPLATE = """你是一個專注於從文章生成高品質問題與答案的 AI 助理，目的是根據文章內容設計出一個繁體中文 QA 對話。請遵循以下指導原則：

### 問題生成原則
1. 問題應與文章內容密切相關，避免提及「文章」或「根據文章」等字眼。
2. 問題需包含明確的關鍵字或關鍵實體（如產品名稱、編號），確保具體且直觀，避免模糊或寬泛的語句。
3. 問題類型應涵蓋以下任務之一: {{ metatags['tasks'] }}
4. 問題必須隨機從文章的不同部分生成或合成，確保全面性與多樣性。
5. 你要模擬客戶的角度，提出有意義且具挑戰性的問題。

---

### 答案生成原則

1. 答案應使用繁體中文，盡可能的簡潔明確，避免冗長敘述。
2. 避免提及「根據文章」等相關字眼，但可以以自然方式傳遞相關資訊來源。
3. 回答需直接回應問題，並提供文章中的關鍵資訊，避免過多延伸。
4. 若文章無法回答某問題，答案應清楚說明「無法找到相關資訊」。

---

### 任務輸出格式
請基於文章內容生成一組 QA，並使用以下格式：
```json
{
    "question": "生成的問題內容",
    "answer": "生成的回答內容"
}
---

### 文章

{{text}}"""
