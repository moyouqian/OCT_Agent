import { Bot, Brain, FlaskConical, MessageCircle, Search } from "lucide-react";

const PROMPT_CARDS = [
  {
    icon: MessageCircle,
    title: "概念问答",
    text: "用通俗的语言解释 OCT 是什么，以及它在生物组织成像中的原理。",
  },
  {
    icon: FlaskConical,
    title: "应变计算",
    text: "对我上传的 phase .mat 文件做矢量法和 CNN 应变估计，并对比结果。",
  },
  {
    icon: Search,
    title: "Deep Research",
    text: "调研 OCT 弹性成像（OCE）近年的主流应变估计方法及其优缺点。",
  },
];

const MEMORY_HINTS = [
  { cmd: "记住", text: "记住 我的实验默认折射率是 1.36" },
  { cmd: "忘记", text: "忘记 我的实验默认折射率" },
  { cmd: "查看记忆", text: "查看记忆" },
];

type Props = {
  onPick: (text: string) => void;
};

export function WelcomeScreen({ onPick }: Props) {
  return (
    <div className="welcome">
      <span className="welcome-icon">
        <Bot size={28} />
      </span>
      <h2>有什么 OCT 相关问题，直接问我</h2>
      <p>
        可以先普通聊天，也可以添加 <code>.mat</code> 文件后让 Agent 做应变计算、联网做深度研究，或检索本地知识库。
      </p>

      <div className="prompt-cards">
        {PROMPT_CARDS.map((card) => {
          const Icon = card.icon;
          return (
            <button type="button" className="prompt-card" key={card.title} onClick={() => onPick(card.text)}>
              <strong>
                <Icon size={16} />
                {card.title}
              </strong>
              <span>{card.text}</span>
            </button>
          );
        })}
      </div>

      <div className="command-hints">
        <span className="command-hint" style={{ cursor: "default" }}>
          <Brain size={13} />
          长期记忆指令
        </span>
        {MEMORY_HINTS.map((hint) => (
          <button type="button" className="command-hint" key={hint.cmd} onClick={() => onPick(hint.text)}>
            <code>{hint.cmd}</code>
          </button>
        ))}
      </div>
    </div>
  );
}
