## placeholders are enclosed by {{{}}}

import re
import json
import random
import logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)-4.7s]  %(message)s")

from transformers import GenerationConfig

default_generation_config = GenerationConfig.from_dict({
    "chat_format": "chatml",
    "eos_token_id": 151643,
    "pad_token_id": 151643,
    "max_window_size": 6144,
    "max_new_tokens": 512,
    "do_sample": True,
    "top_k": 0,
    "top_p": 0.8,
    "repetition_penalty": 1.1,
})

class SampleAgent(object):
    def __init__(self, model, tokenizer, action_selection_max_retries=3):
        self.model = model
        self.tokenizer = tokenizer
        self.action_selection_max_retries = action_selection_max_retries
        self.should_exit = False
        self.action_selecting_prompt_template = """你是一个智能体，能够使用以下工具，完成{{{ Task Name }}}任务：
- [{{{ Tool A Name }}}]：{{{ Tool A Documentation }}}。输入：{{{ Tool A Input Params }}}，输出：{{{ Tool A Response Properties }}}。
- [{{{ Tool B Name }}}]：{{{ Tool B Documentation }}}。输入：{{{ Tool B Input Params }}}，输出：{{{ Tool B Response Properties }}}。
- [{{{ Tool C Name }}}]：{{{ Tool C Documentation }}}。输入：{{{ Tool C Input Params }}}，输出：{{{ Tool C Response Properties }}}。
- [{{{ Tool D Name }}}]：{{{ Tool D Documentation }}}。输入：{{{ Tool D Input Params }}}，输出：{{{ Tool D Response Properties }}}。

你的工作流程如下：
1. 使用[{{{ Tool A Name }}}]，获取xxx信息，成功后跳转至2。
  1.1 如果[{{{ Tool A Name }}}]查询失败，则调用[{{{ Tool C Name }}}]，获取xxx信息，成功后跳转至2。
    1.1.1 如果[{{{ Tool C Name }}}]调用失败，则调用[{{{ Tool D Name }}}]，获取yyy信息，成功后跳转至3。
      1.1.1.1 如果[{{{ Tool D Name }}}]调用失败，则任务执行失败，直接回答问题（[Answer-with-Failure]）。
2. 已知xxx时使用[{{{ Tool B Name }}}]，根据xxx获取yyy，成功后跳转至3。
  2.1 如果[{{{ Tool B Name }}}]查询失败，则调用[{{{ Tool D Name }}}]，获取yyy信息，成功后跳转至3。
    2.1.1 如果[{{{ Tool D Name }}}]调用失败，则任务执行失败，直接回答问题（[Answer-with-Failure]）。
3. 已知yyy时用yyy回答问题（[Answer-with-Success]）。

参考以上工作流程，你需要根据[问题(Question)]和[已知信息（Observation）]判断当前步骤处于工作流程中的哪一步，根据工作流程选择下一步应该执行的[动作（Action）]，并为选择的动作提供正确的**JSON**形式的[输入参数（Action_Input）]。

[示例]

Question: {{{ ICL Question 1 }}}
Observation: {}
Thoughts: 没有任何已知信息，此时为初始状态。所以根据工作流程中“1. 使用[{{{ Tool A Name }}}]，获取xxx信息”，接下来应该使用[{{{ Tool A Name }}}]。
Action: [{{{ Tool A Name }}}]
Action_Input: {"param1": value1, "param2": value2}

Question: {{{ ICL Question 2 }}}
Observation: {"oaram1": value1, "param2": value2, "xxx1": xxxvalue1, "xxx2": xxxvalue2}
Thoughts: xxx已知，所以根据工作流程“2. 已知xxx时使用[{{{ Tool B Name }}}]，根据xxx获取yyy”，接下来应该使用[{{{ Tool B Name }}}]。
Action: [{{{ Tool B Name }}}]
Action_Input: {"oaram1": value1, "param2": value2, "xxx1": xxxvalue1, "xxx2": xxxvalue2}

Question: {{{ ICL Question 3 }}}
Observation: {"yyy": yyyvalue}
Thoughts: yyy已知，所以根据工作流程“3. 已知yyy时用yyy回答问题（[Answer-with-Success]）”，接下来应该回答用户问题（[Answer-with-Success]）。
Action: [Answer-with-Success]
Action_Input: {"question": "{{{ ICL Question 3 }}}", "yyy": yyyvalue}

Question: {{{ ICL Question 4 }}}
Observation: {"param1": value1, "param2": value2, "tool_a_status": "error"}
Thoughts: 已知[{{{ Tool A Name }}}]查询失败，所以根据工作流程“1.1 如果[{{{ Tool A Name }}}]查询失败，则调用[{{{ Tool C Name }}}]”，接下来应该使用[{{{ Tool C Name }}}]查询xxx。
Action: [{{{ Tool C Name }}}]
Action_Input: {"param1": value1, "param2": value2}

Question: {{{ ICL Question 5 }}}
Observation: {"param1": value1, "param2": value2, "tool_b_status": "error"}
Thoughts: 已知[{{{ Tool B Name }}}]查询失败，所以根据工作流程“2.1 如果[{{{ Tool B Name }}}]查询失败，则调用[{{{ Tool D Name }}}]”，接下来应该使用[{{{ Tool D Name }}}]查询yyy。
Action: [{{{ Tool D Name }}}]
Action_Input: {"param1": value1, "param2": value2}

Question: {{{ ICL Question 6 }}}
Observation: {"param1": value1, "param2": value2, "tool_c_status": "error"}
Thoughts: 已知[{{{ Tool C Name }}}]查询失败，所以根据工作流程“1.1.1 如果[{{{ Tool C Name }}}]调用失败，则调用[{{{ Tool D Name }}}]”，接下来应该调用[{{{ Tool D Name }}}]查询yyy。
Action: [{{{ Tool D Name }}}]
Action_Input: {"param1": value1, "param2": value2}

Question: {{{ ICL Question 7 }}}
Observation: {"param1": value1, "param2": value2, "tool_d_status": "error"}
Thoughts: 已知[{{{ Tool D Name }}}]查询失败，所以根据工作流程1.1.1.1和2.1.1中的“如果[{{{ Tool D Name }}}]调用失败，则任务执行失败”，接下来应该回答用户问题（[Answer-with-Failure]）。
Action: [Answer-with-Failure]
Action_Input: {"question": "{{{ ICL Question 7 }}}"}

请你参考以上示例，为下面的[问题(Question)]和[已知信息（Observation）]选择合理的下一步[动作（Action）]和[输入参数（Action_Input）]。

Question: {{% question %}}
Observation: {{% observation %}}
"""
    
    def action_selection(self, question, observation):

        def parse_action_answer(_answer_string):
            _rep_answer_string = _answer_string.replace("：", ": ")
            action_separator = "Action:"
            action_input_separator = "Action_Input:"
            if action_separator not in _answer_string or action_input_separator not in _answer_string:
                logging.info("Separator Missing in LLM response.")
                return None, None, None
            action_index = _rep_answer_string.index(action_separator)
            action_input_index = _rep_answer_string.index(action_input_separator)
            if action_index < action_input_index:
                action_input_content = _rep_answer_string[(action_input_index + len(action_input_separator)):].strip()
                action_content = _rep_answer_string[(action_index + len(action_separator)):action_input_index].strip()
                thoughts_content = _rep_answer_string[:action_index].strip()
            else:
                action_content = _rep_answer_string[(action_index + len(action_separator)):].strip()
                action_input_content = _rep_answer_string[(action_input_index + len(action_input_separator)):action_index].strip()
                thoughts_content = _rep_answer_string[:action_input_index].strip()
            action_name = action_content.replace('[', '').replace(']', '').replace('-', '_').replace(' ', '_').lower()
            try:
                action_func = getattr(self, action_name)
            except NameError as e:
                logging.error(f"Unknown action: {action_name}; ERROR: {repr(e)}")
                return None, None, None
            try:
                if "\n" in action_input_content:
                    action_input_content = action_input_content.split("\n")[0]
                action_input_dict = json.loads(action_input_content.replace("'", "\"")) 
            except Exception as e:
                logging.error(f"JSON decoding for action_input_dict failed: {action_input_content}; ERROR: {repr(e)}")
                return None, None, None
            return action_func, action_input_dict, thoughts_content

        # use LLM to select actions
        logging.info(f"[Action Selection] START")
        logging.info(f"[Action Selection] Observation = {observation}")
        prompt = self.action_selecting_prompt_template.replace("{{% question %}}", question)\
            .replace("{{% observation %}}", str(observation))
        action_selection_llm_answer, _ = self.model.chat(self.tokenizer, prompt, system="你是一个乐于助人的助手。", \
            history=None, generation_config=default_generation_config)
        action_func, action_input_dict, thoughts_content = parse_action_answer(action_selection_llm_answer)
        logging.info(f"[Action Selection] {thoughts_content}")

        retry = 0
        while action_func is None and retry < self.action_selection_max_retries:
            # TODO: Add reflection prompts here
            retry += 1
            logging.info(f"[Action Selection] Parsing LLM response failed. Retry generating (Retry {retry}) ...")
            action_selection_llm_answer, _ = self.model.chat(self.tokenizer, prompt, system="你是一个乐于助人的助手。", \
                history=None, generation_config=default_generation_config)
            action_selection_llm_answer = action_selection_llm_answer.replace("\n", "\n----------\n")
            logging.info(f"[Action Selection] LLM Response: \n{action_selection_llm_answer}")
            action_func, action_input_dict, thoughts_content = parse_action_answer(action_selection_llm_answer)
            logging.info(f"[Action Selection] {thoughts_content}" if thoughts_content is not None else "[Action Selection] Thoughts content: None")
        logging.info(f"[Action Selection] -----> [{action_func.__name__.upper().replace('_', '-')}], input = {action_input_dict}")
        
        return action_func, action_input_dict
    
    def tool_a_action(self, param1, param2, **kwargs):
        logging.info("[{{{Tool A Name}}}] START")
        try:
            logging.info(f"[{{{{{{Tool A Name}}}}}}] param1 = {param1}, param2 = {param2}")
            ### tool_a_function is implemented somewhere else
            xxxvalue1, xxxvalue2 = tool_a_function(param1, param2)
            logging.info(f"[{{{{{{Tool A Name}}}}}}] Reponse xxx1 = {xxxvalue1}, xxx2 = {xxxvalue2}")
            # This observation will be passed to tool_b.
            # Here we need to return param1 and param2, because tool_b may encouter error.
            # Under this case, the tool_d will be called, whose input is param1 and param2
            # so, param1 and param2 should also be passed to tool_b
            return {
                "param1": param1,
                "param2": param2,
                "xxx1": xxxvalue1,
                "xxx2": xxxvalue2
            }
        except Exception as e:
            logging.error(f"[{{{{{{Tool A Name}}}}}}] ERROR: {repr(e)}")
            return {
                "param1": param1,
                "param2": param2,
                "tool_a_status": "error"
            }
    
    def tool_b_action(self, xxx1, xxx2, **kwargs):
        logging.info("[{{{Tool B Name}}}] START")
        try:
            logging.info(f"[{{{{{{Tool B Name}}}}}}] xxx1 = {xxx1}, xxx2 = {xxx2}")
            ### tool_b_function is implemented somewhere else
            yyy = tool_b_function(xxx1, xxx2)
            logging.info(f"[{{{{{{Tool B Name}}}}}}] Reponse yyy = {yyy}")
            return {
                "yyy": yyy
            }
        except Exception as e:
            logging.error(f"[{{{{{{Tool B Name}}}}}}] ERROR: {repr(e)}")
            kw_clone = {k: v for k, v in kwargs.items()}
            kw_clone.update({"tool_b_status": "error"})
            return kw_clone
    
    def tool_c_action(self, param1, param2, **kwargs):
        logging.info("[{{{Tool C Name}}}] START")
        try:
            logging.info(f"[{{{{{{Tool C Name}}}}}}] param1 = {param1}, param2 = {param2}")
            ### tool_c_function is implemented somewhere else
            xxxvalue1, xxxvalue2 = tool_c_function(param1, param2)
            logging.info(f"[{{{{{{Tool C Name}}}}}}] Reponse xxx1 = {xxxvalue1}, xxx2 = {xxxvalue2}")
            return {
                "param1": param1,
                "param2": param2,
                "xxx1": xxxvalue1,
                "xxx2": xxxvalue2
            }
        except Exception as e:
            logging.error(f"[{{{{{{Tool C Name}}}}}}] ERROR: {repr(e)}")
            return {
                "param1": param1,
                "param2": param2,
                "tool_b_status": "error"
            }
    
    def tool_d_action(self, param1, param2, **kwargs):
        logging.info("[{{{Tool D Name}}}] START")
        try:
            logging.info(f"[{{{{{{Tool D Name}}}}}}] param1 = {param1}, param2 = {param2}")
            ### tool_d_function is implemented somewhere else
            yyy = tool_d_function(param1, param2)
            logging.info(f"[{{{{{{Tool D Name}}}}}}] Reponse yyy = {yyy}")
            return {
                "yyy": yyy
            }
        except Exception as e:
            logging.error(f"[{{{{{{Tool D Name}}}}}}] ERROR: {repr(e)}")
            return {
                "param1": param1,
                "param2": param2,
                "tool_d_status": "error"
            }

    def answer_with_success(self, question, yyy, **kwargs):
        logging.info("[Answer-with-Success] START")
        ### construct_prompt_with_yyy is implemented somewhere else
        prompt = construct_prompt_with_yyy(question, yyy)
        response, _ = self.model.chat(self.tokenizer, prompt, system="你是一个乐于助人的助手。", \
            history=None, generation_config=default_generation_config)
        self.should_exit = True
        return response
    
    def answer_with_failure(self, question, **kwargs):
        logging.info("[Answer-with-Failure] START")
        response, _ = self.model.chat(self.tokenizer, question, system="你是一个乐于助人的助手。", \
            history=None, generation_config=default_generation_config)
        self.should_exit = True
        return response
    
    def run(self, question):
        observation = {}
        while not self.should_exit:
            try:
                action_func, action_input_dict = self.action_selection(question, observation)
                assert action_func is not None, "action_selection result in None"
            except Exception as e:
                logging.info(f"[Action selection] ERROR: {repr(e)}")
                observation = self.answer_with_failure(question)
                break
            observation = action_func(**action_input_dict)
        self.should_exit = False
        final_response = observation
        print("-----------------------")
        print(final_response)

