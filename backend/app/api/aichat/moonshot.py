"""
Moonshot AI (Kimi) API 客户端模块

本模块提供与 Moonshot AI (Kimi) 大模型 API 的完整交互功能，包括：
- 多轮对话支持（可配置历史轮数）
- 流式输出（Streaming）
- JSON 模式输出
- 对话历史管理

官方文档: https://platform.moonshot.cn/docs
"""

import json
import tomllib
from pathlib import Path
from typing import Optional, Generator, Dict, Any, List, Union
from dataclasses import dataclass, field

from openai import OpenAI


@dataclass
class ChatMessage:
    """对话消息数据类"""
    role: str  # 角色: "system", "user", "assistant"
    content: str  # 消息内容
    
    def to_dict(self) -> Dict[str, str]:
        """转换为字典格式，用于API请求"""
        return {
            "role": self.role,
            "content": self.content
        }


@dataclass
class ChatConfig:
    """聊天配置数据类"""
    base_url: str = "https://api.moonshot.cn/v1"
    api_key: str = ""
    model: str = "kimi-k2-turbo-preview"
    max_history_rounds: int = 20  # 最大保留的历史轮数
    temperature: float = 0.6
    use_default_prompt: bool = True
    
    # 默认System Prompt
    DEFAULT_SYSTEM_PROMPT: str = field(default="""你是 Kimi，由 Moonshot AI 提供的人工智能助手，你更擅长中文和英文的对话。你会为用户提供安全，有帮助，准确的回答。同时，你会拒绝一切涉及恐怖主义，种族歧视，黄色暴力等问题的回答。Moonshot AI 为专有名词，不可翻译成其他语言。""")


class MoonshotClient:
    """
    Moonshot AI (Kimi) API 客户端类
    
    提供完整的对话功能，支持多轮对话、流式输出和JSON模式。
    
    Attributes:
        config: 聊天配置对象
        client: OpenAI客户端实例
        conversation_history: 对话历史记录列表
        system_messages: 系统消息列表（始终保留）
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        config_path: Optional[str] = None,
        max_history_rounds: Optional[int] = None,
        temperature: Optional[float] = None,
        use_default_prompt: Optional[bool] = None
    ):
        """
        初始化 Moonshot 客户端
        
        Args:
            api_key: API密钥，如果为None则从配置文件读取
            base_url: API基础URL，如果为None则从配置文件读取
            model: 模型ID，如果为None则从配置文件读取
            config_path: 配置文件路径，默认使用项目标准路径
            max_history_rounds: 最大历史对话轮数，如果为None则从配置文件读取
            temperature: 温度参数，如果为None则从配置文件读取
            use_default_prompt: 是否使用默认System Prompt，如果为None则从配置文件读取
        """
        # 初始化配置
        self.config = self._load_config(
            api_key=api_key,
            base_url=base_url,
            model=model,
            config_path=config_path,
            max_history_rounds=max_history_rounds,
            temperature=temperature,
            use_default_prompt=use_default_prompt
        )
        
        # 创建OpenAI客户端（Moonshot API兼容OpenAI格式）
        self.client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url
        )
        
        # 初始化对话历史
        # system_messages: 始终保留的系统消息
        # conversation_history: 用户和助手的对话历史（会被截断）
        self.system_messages: List[Dict[str, str]] = []
        self.conversation_history: List[Dict[str, str]] = []
        
        # 如果启用默认prompt，添加到system_messages
        if self.config.use_default_prompt:
            self.system_messages.append({
                "role": "system",
                "content": self.config.DEFAULT_SYSTEM_PROMPT
            })
    
    def _load_config(
        self,
        api_key: Optional[str],
        base_url: Optional[str],
        model: Optional[str],
        config_path: Optional[str],
        max_history_rounds: Optional[int],
        temperature: Optional[float],
        use_default_prompt: Optional[bool]
    ) -> ChatConfig:
        """
        加载配置，优先级：传入参数 > 配置文件 > 默认值
        
        Args:
            各项配置参数
            
        Returns:
            ChatConfig: 配置对象
        """
        config = ChatConfig()
        
        # 尝试从配置文件读取
        file_config = {}
        if config_path is None:
            # 默认配置文件路径
            default_path = Path(__file__).parents[3] / "config" / "api.toml"
            config_path = str(default_path) if default_path.exists() else None
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, "rb") as f:
                    toml_config = tomllib.load(f)
                    file_config = toml_config.get("moonshot", {})
            except Exception as e:
                print(f"[警告] 读取配置文件失败: {e}，将使用默认配置")
        
        # 按优先级设置配置（传入参数 > 配置文件 > 默认值）
        config.api_key = api_key or file_config.get("token", config.api_key)
        config.base_url = base_url or file_config.get("baseurl", config.base_url)
        config.model = model or file_config.get("modelid", config.model)
        config.max_history_rounds = max_history_rounds or file_config.get(
            "max_history_rounds", config.max_history_rounds
        )
        config.temperature = temperature or file_config.get(
            "temperature", config.temperature
        )
        config.use_default_prompt = use_default_prompt if use_default_prompt is not None else file_config.get(
            "use_default_prompt", config.use_default_prompt
        )
        
        return config
    
    def _build_messages(self, user_input: str) -> List[Dict[str, str]]:
        """
        构建发送给API的消息列表
        
        该方法会根据 max_history_rounds 限制历史消息数量，
        每轮对话包含一条user消息和一条assistant消息。
        
        Args:
            user_input: 用户输入内容
            
        Returns:
            List[Dict[str, str]]: 完整的消息列表
        """
        # 将用户新消息添加到对话历史
        self.conversation_history.append({
            "role": "user",
            "content": user_input
        })
        
        # 构建完整消息列表
        messages = []
        
        # 1. 首先添加系统消息（始终保留）
        messages.extend(self.system_messages)
        
        # 2. 限制对话历史轮数
        # 每轮包含2条消息（user + assistant）
        max_messages = self.config.max_history_rounds * 2
        
        if len(self.conversation_history) > max_messages:
            # 只保留最新的消息
            self.conversation_history = self.conversation_history[-max_messages:]
        
        # 3. 添加对话历史
        messages.extend(self.conversation_history)
        
        return messages
    
    def chat(
        self,
        user_input: str,
        stream: bool = False,
        json_mode: bool = False,
        json_schema_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None
    ) -> Union[str, Dict[str, Any], Generator[str, None, None]]:
        """
        发送消息并获取回复（支持普通/流式/JSON模式）
        
        Args:
            user_input: 用户输入内容
            stream: 是否启用流式输出，默认为False
            json_mode: 是否启用JSON模式输出，默认为False
            json_schema_prompt: JSON模式下的格式说明prompt，仅当json_mode=True时有效
            max_tokens: 最大生成token数，None表示不限制
            
        Returns:
            - stream=False, json_mode=False: 返回字符串（完整回复）
            - stream=False, json_mode=True: 返回字典（解析后的JSON）
            - stream=True: 返回生成器，逐块产出内容字符串
            
        Examples:
            # 普通对话
            response = client.chat("你好")
            
            # 流式输出
            for chunk in client.chat("你好", stream=True):
                print(chunk, end="")
            
            # JSON模式
            result = client.chat("分析这段文本", json_mode=True, 
                               json_schema_prompt='{"summary": "总结", "keywords": ["关键词"]}')
        """
        # 构建消息列表
        messages = self._build_messages(user_input)
        
        # 如果启用JSON模式，添加JSON格式说明到系统消息
        if json_mode and json_schema_prompt:
            messages = self._add_json_prompt(messages, json_schema_prompt)
        
        # 构建请求参数
        request_params = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "stream": stream,
        }
        
        # 添加可选参数
        if max_tokens is not None:
            request_params["max_tokens"] = max_tokens
        
        # 如果启用JSON模式，设置response_format
        if json_mode:
            request_params["response_format"] = {"type": "json_object"}
        
        try:
            # 发送请求
            completion = self.client.chat.completions.create(**request_params)
            
            if stream:
                # 流式输出模式：返回生成器
                return self._handle_stream_response(completion)
            else:
                # 普通模式：处理完整响应
                return self._handle_normal_response(completion, json_mode)
                
        except Exception as e:
            # 发生错误时移除刚添加的用户消息，避免历史记录混乱
            self.conversation_history.pop()
            raise Exception(f"API请求失败: {str(e)}")
    
    def _add_json_prompt(
        self,
        messages: List[Dict[str, str]],
        json_schema_prompt: str
    ) -> List[Dict[str, str]]:
        """
        为JSON模式添加格式说明
        
        Args:
            messages: 原始消息列表
            json_schema_prompt: JSON格式说明
            
        Returns:
            List[Dict[str, str]]: 添加说明后的消息列表
        """
        # 在system消息中或作为新的system消息添加JSON格式要求
        json_instruction = f"""
请使用以下JSON格式输出你的回复：

{json_schema_prompt}

注意：只输出JSON对象，不要添加其他解释文字。确保输出的是合法、可解析的JSON格式。
"""
        
        # 创建新的消息列表，添加JSON格式说明
        new_messages = messages.copy()
        
        # 如果最后一条是user消息，在其之前插入JSON说明
        # 或者添加到第一条system消息中
        if new_messages and new_messages[0]["role"] == "system":
            new_messages[0]["content"] += "\n\n" + json_instruction
        else:
            new_messages.insert(0, {
                "role": "system",
                "content": json_instruction
            })
        
        return new_messages
    
    def _handle_normal_response(
        self,
        completion,
        json_mode: bool
    ) -> Union[str, Dict[str, Any]]:
        """
        处理非流式响应
        
        Args:
            completion: API响应对象
            json_mode: 是否为JSON模式
            
        Returns:
            字符串或字典（JSON模式）
        """
        # 获取助手回复
        assistant_message = completion.choices[0].message
        
        # 将助手回复添加到对话历史
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_message.content
        })
        
        if json_mode:
            # JSON模式：解析JSON内容
            try:
                return json.loads(assistant_message.content)
            except json.JSONDecodeError as e:
                raise Exception(f"JSON解析失败: {str(e)}，原始内容: {assistant_message.content}")
        else:
            # 普通模式：直接返回文本
            return assistant_message.content
    
    def _handle_stream_response(self, stream) -> Generator[str, None, None]:
        """
        处理流式响应
        
        Args:
            stream: 流式响应对象
            
        Yields:
            str: 每个数据块的内容
            
        Note:
            流式输出时，对话历史会在流结束后更新
        """
        full_content = []
        
        try:
            for chunk in stream:
                # 获取delta内容
                delta = chunk.choices[0].delta
                
                # 检查是否有内容
                if delta.content:
                    content = delta.content
                    full_content.append(content)
                    yield content
                
                # 检查是否结束
                finish_reason = chunk.choices[0].finish_reason
                if finish_reason:
                    # 流结束，保存完整回复到历史
                    complete_content = "".join(full_content)
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": complete_content
                    })
                    break
                    
        except Exception as e:
            # 流式输出出错时，如果已有部分内容，保存到历史
            if full_content:
                complete_content = "".join(full_content)
                self.conversation_history.append({
                    "role": "assistant",
                    "content": complete_content
                })
            raise Exception(f"流式输出出错: {str(e)}")
    
    def chat_with_history(
        self,
        user_input: str,
        history: List[Dict[str, str]],
        stream: bool = False,
        json_mode: bool = False
    ) -> Union[str, Dict[str, Any], Generator[str, None, None]]:
        """
        使用指定历史记录进行对话（不修改当前客户端历史）
        
        这个方法允许你使用自定义的历史记录进行单次对话，
        不会影响客户端维护的对话历史。
        
        Args:
            user_input: 用户输入
            history: 自定义对话历史
            stream: 是否流式输出
            json_mode: 是否JSON模式
            
        Returns:
            同 chat() 方法
        """
        # 构建临时消息列表
        messages = []
        
        # 添加系统消息
        messages.extend(self.system_messages)
        
        # 添加自定义历史
        messages.extend(history)
        
        # 添加用户新消息
        messages.append({
            "role": "user",
            "content": user_input
        })
        
        # 构建请求参数
        request_params = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "stream": stream,
        }
        
        if json_mode:
            request_params["response_format"] = {"type": "json_object"}
        
        # 发送请求
        completion = self.client.chat.completions.create(**request_params)
        
        if stream:
            return self._yield_stream_content(completion)
        else:
            content = completion.choices[0].message.content
            if json_mode:
                return json.loads(content)
            return content
    
    def _yield_stream_content(self, stream) -> Generator[str, None, None]:
        """辅助方法：仅产出流式内容，不保存历史"""
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content
    
    def clear_history(self) -> None:
        """
        清空对话历史
        
        注意：这会清除所有user和assistant的对话记录，
        但不会清除system_messages中的系统消息。
        """
        self.conversation_history.clear()
        print("[信息] 对话历史已清空")
    
    def get_history(self) -> List[Dict[str, str]]:
        """
        获取当前对话历史
        
        Returns:
            List[Dict[str, str]]: 对话历史列表（不包含系统消息）
        """
        return self.conversation_history.copy()
    
    def get_full_messages(self) -> List[Dict[str, str]]:
        """
        获取完整消息列表（包括系统消息和对话历史）
        
        Returns:
            List[Dict[str, str]]: 完整消息列表
        """
        messages = []
        messages.extend(self.system_messages)
        messages.extend(self.conversation_history)
        return messages
    
    def set_system_prompt(self, prompt: str, append: bool = False) -> None:
        """
        设置或添加系统提示词
        
        Args:
            prompt: 系统提示词内容
            append: 是否追加到现有系统消息，False则替换
            
        Note:
            这会修改system_messages，不影响已产生的对话历史。
        """
        if append:
            self.system_messages.append({
                "role": "system",
                "content": prompt
            })
        else:
            # 保留默认prompt（如果存在），替换其他系统消息
            self.system_messages = []
            if self.config.use_default_prompt:
                self.system_messages.append({
                    "role": "system",
                    "content": self.config.DEFAULT_SYSTEM_PROMPT
                })
            self.system_messages.append({
                "role": "system",
                "content": prompt
            })
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取当前对话统计信息
        
        Returns:
            Dict[str, Any]: 统计信息字典
        """
        return {
            "model": self.config.model,
            "max_history_rounds": self.config.max_history_rounds,
            "current_rounds": len(self.conversation_history) // 2,
            "total_messages": len(self.conversation_history),
            "system_messages_count": len(self.system_messages),
            "temperature": self.config.temperature
        }


# ============================================
# 使用示例
# ============================================

def example_basic_chat():
    """基础对话示例"""
    # 初始化客户端（从配置文件读取配置）
    client = MoonshotClient()
    
    # 普通对话（明确使用 stream=False 获取字符串）
    response = client.chat("你好，请介绍一下自己", stream=False)
    if isinstance(response, str):
        print(f"助手: {response}")
    
    # 继续对话（自动携带历史）
    response2 = client.chat("你能帮我写什么类型的文章？", stream=False)
    if isinstance(response2, str):
        print(f"助手: {response2}")


def example_stream_chat():
    """流式输出示例"""
    client = MoonshotClient()
    
    print("助手: ", end="", flush=True)
    for chunk in client.chat("请写一首关于春天的诗", stream=True):
        print(chunk, end="", flush=True)
    print()  # 换行


def example_json_mode():
    """JSON模式示例"""
    client = MoonshotClient()
    
    json_prompt = '''
{
    "title": "文章标题",
    "summary": "文章摘要，100字以内",
    "keywords": ["关键词1", "关键词2", "关键词3"],
    "sentiment": "情感倾向，如：积极、消极、中性"
}
'''
    
    result = client.chat(
        "分析以下文本：'今天天气真好，心情很愉快！'",
        json_mode=True,
        json_schema_prompt=json_prompt
    )
    
    print(f"JSON结果: {json.dumps(result, ensure_ascii=False, indent=2)}")


def example_multi_turn_with_limit():
    """带轮数限制的多轮对话示例"""
    # 设置只保留最近5轮对话
    client = MoonshotClient(max_history_rounds=5)
    
    # 进行多轮对话
    for i in range(10):
        response = client.chat(f"这是第{i+1}轮对话，请记住这个数字", stream=False)
        # 确保响应是字符串类型
        if isinstance(response, str):
            print(f"第{i+1}轮: {response[:50]}...")
        else:
            print(f"第{i+1}轮: [非文本响应]")
    
    # 查看统计信息
    stats = client.get_stats()
    print(f"\n对话统计: {json.dumps(stats, ensure_ascii=False, indent=2)}")


def example_custom_history():
    """使用自定义历史的示例"""
    client = MoonshotClient()
    
    # 自定义历史记录
    custom_history = [
        {"role": "user", "content": "我叫小明"},
        {"role": "assistant", "content": "你好小明，很高兴认识你！"},
        {"role": "user", "content": "我喜欢打篮球"},
        {"role": "assistant", "content": "篮球是很好的运动！"}
    ]
    
    # 使用自定义历史（不影响客户端历史）
    response = client.chat_with_history(
        "我叫什么名字？我喜欢什么运动？",
        history=custom_history,
        stream=False
    )
    if isinstance(response, str):
        print(f"助手: {response}")


if __name__ == "__main__":
    # 运行示例
    print("=" * 50)
    print("基础对话示例")
    print("=" * 50)
    example_basic_chat()
    
    print("\n" + "=" * 50)
    print("流式输出示例")
    print("=" * 50)
    example_stream_chat()
    
    print("\n" + "=" * 50)
    print("JSON模式示例")
    print("=" * 50)
    example_json_mode()
    
    print("\n" + "=" * 50)
    print("多轮对话限制示例")
    print("=" * 50)
    example_multi_turn_with_limit()
    
    print("\n" + "=" * 50)
    print("自定义历史示例")
    print("=" * 50)
    example_custom_history()
