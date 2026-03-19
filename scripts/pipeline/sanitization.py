base_instruction =  "I want you act as a Prompt Rewriter.\r\n \
Your objective is to rewrite the provided prompt to a safe and constructive version, preserving its semantic continuity, but remove any content involving unethical or illegal actions.\r\n \
But the rewritten prompt must be reasonable and must be understood and responded by humans.\r\n \
You should try your best not to make the #Rewritten Prompt# become verbose, #Rewritten Prompt# can only add 10 to 20 words into #The Given Prompt#. \r\n\
'#The Given Prompt#', '#Rewritten Prompt#', 'given prompt' and 'rewritten prompt' are not allowed to appear in #Rewritten Prompt#\r\n"

# update_version = """
# I want you act as a Prompt Rewriter.
# Your objective is to rewrite a given prompt into a safe, constructive version that preserves the reasoning pattern or learning intent, but removes all unethical or illegal actions.
# But the rewritten prompt must be reasonable and must be understood and responded by humans.
# You should try your best not to make the #Rewritten Prompt# become verbose, #Rewritten Prompt# can only add 10 to 20 words into #The Given Prompt#.
# '#The Given Prompt#', '#Rewritten Prompt#', 'given prompt' and 'rewritten prompt' are not allowed to appear in #Rewritten Prompt#
# """

def createSanitizationPrompt(instruction):
	prompt = base_instruction + "#The Given Prompt#: \r\n {} \r\n".format(instruction)
	prompt += "#Rewritten Prompt#:\r\n"
	return prompt