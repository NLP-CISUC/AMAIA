import os
import slack
import asyncio
import re

import controller
import ast

'''
Slack app controler, responsible for back-end of the app
'''

# constants
SLACK_SIGNING_SECRET = os.environ['SLACK_SIGNING_SECRET']
SLACK_BOT_TOKEN = os.environ['COBAIA_BOT_TOKEN']

# cobaiabot's user ID in Slack: value is assigned after the bot starts up
bot_id = None

# constants
MENTION_REGEX = "^<@(|[WU].+?)>(.*)"

# Function that allows the interface to communicate with Slack
@slack.RTMClient.run_on(event='message')
async def handle_message(**payload):

    data = payload['data']

    # get the text of the message in order to search for a response
    text = data.get('text')
    # get the channel in which the message was sent
    channel = data.get('channel')

	# if the first letter in the channel name is a D, it means that is was a direct message to the chatbot
    if channel[0] == "D":
        if data.get('subtype') is None and text:
            message = controller.web_chat_interface(text)
            slack_message = message.replace('<br>', '\n')

            webclient = payload['web_client']
            await webclient.chat_postMessage(
                channel=channel,
                text=slack_message
            )
    else:
        # if not a direct message, check whether or not the chatbot was mentioned in the conversation
        user_id, text = parse_direct_mention(text)

        if user_id == bot_id:
            if data.get('subtype') is None and text:
                message = controller.web_chat_interface(text)
                slack_message = message.replace('<br>', '\n')

                webclient = payload['web_client']
                await webclient.chat_postMessage(
                    channel=channel,
                    text=slack_message
                )

# Function to parse a direct mention of the bot in Slack
def parse_direct_mention(message_text):
	"""
		Finds a direct mention (a mention that is at the beginning) in message text
		and returns the user ID which was mentioned. If there is no direct mention, returns None
	"""
	matches = re.search(MENTION_REGEX, message_text)
	# the first group contains the username, the second group contains the remaining message
	return (matches.group(1), matches.group(2).strip()) if matches else (None, None)

def load_config(config_file):
    lines = {}
    with open(config_file,"r") as f:
        contents = f.read()
        lines = ast.literal_eval(contents)
        f.close()
    return lines

if __name__ == '__main__':
    #Loading models and starting app
    configs = load_config("config.txt")
    controller.start_controler(
        configs["agents"], configs["corpus"], configs["out_of_domain_corpus"],
        configs["out_of_domain_classifier"], configs["number_of_answers_per_agent"], configs["decision"],
        configs["theta"], configs["w2v_embeddings"], configs["bert_embeddings"])

    slack_web_client = slack.WebClient(SLACK_BOT_TOKEN)

    # extract the bot_id in order to check if it is mentioned in a channel conversation
    if bot_id is None:
        bot_id = slack_web_client.api_call("auth.test")["user_id"]

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    slack_client = slack.RTMClient(
        token=SLACK_BOT_TOKEN, run_async=True, loop=loop
    )
    loop.run_until_complete(slack_client.start())
