import openai

apikey = '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'
def chat(input):
    openai.api_key = apikey
    completion = openai.ChatCompletion.create(
        model = 'gpt-3.5-turbo',
        message = [{'role':'assistant','content':input}]
    )
    return completion.choices[0].message.content

while 1:
    print(chat(input('请输入指令：')))