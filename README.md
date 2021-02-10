# AMAIA – agente pergunta-reposta

O AMAIA é uma plataforma de agentes baseados em perguntas e respostas, fornecidas num ficheiro de texto.
A adaptação a diferentes domínios requer apenas a utilização de diferentes ficheiros de perguntas e respostas. 
Concretamente, o AMAIA foi aplicado a um conjunto de serviços da Administração Pública.
Aqui, utiliza o corpus [AIA-BDE](https://github.com/NLP-CISUC/AIA-BDE) para perguntas sobre o domínio, e um corpus de chitchat para responder a interações fora do domínio e simular conversas.

A versão mais recente do sistema é descrita no seguinte artigo científico:
<pre>
@article{santos_etal:information2020,
	author = {Jos{\'e} Santos and Lu{\'\i}s Duarte and Jo{\~a}o Ferreira and Ana Alves and Hugo {Gon{\c c}alo~Oliveira}},
	journal = {Information},
	month = {September},
	number = {9},
	title = {Developing {A}maia: A Conversational Agent for Helping {P}ortuguese Entrepreneurs -- {A}n Extensive Exploration of Question-Matching Approaches for {P}ortuguese},
	volume = {11},
	year = {2020}}
</pre>

##	Instalação
Para instalar o AMAIA é necessário copiar os ficheiros para o diretório desejado, e instalar os requisitos através da linha de comandos:

> pip3 install -r requirements.txt --no-cache-dir

Para utilizar o agente "STS" será necessário fazer o download dos modelos de word embeddings, procedimento que poderá demorar alguns minutos. Para tal basta correr os seguintes comandos no interpretador do Python no terminal:

>>> import ASAPPpy

>>> ASAPPy.download()

Depois disto, o AMAIA estará pronto a correr (ver secção 3).

## Configuração

Existem várias configurações possíveis para o AMAIA, de forma a permitir que este se adapte a diversas situações. Para alterar os mesmos, é necessário alterar o ficheiro config.txt, onde se podem alterar as seguintes opções:

	agents: tem como valor uma lista com identificadores de agentes. Se a lista tiver mais do que um identificador, os respetivos agentes funcionarão em paralelo e será usada uma estratégia de decisão para escolher a melhor resposta a dar (ver opção decision).
Estão disponíveis os agentes com os seguintes identificadores: "Whoosh", "Bert", "W2V", "STS".

	corpus: os agentes respondem a interações com base em ficheiros de texto com perguntas (linhas começadas por P:) e respostas (linhas começadas por R:). Os ficheiros a usar devem ser colocados numa lista dada como valor desta opção.

	out_of_domain_corpus: os agentes podem usar um corpo diferente para responder a perguntas consideradas do domínio e para perguntas fora do domínio. A localização do segundo corpo deve ser colocada como valor desta opção. Quando esta opção não está vazio, um classificador (ver opção out_of_domain_classifier) será usado para marcar cada interação como domínio ou fora do domínio e obter a resposta a partir do corpo adequado.
Se este parâmetro estiver vazio, todas as respostas serão obtidas a partir dos corpos na opção corpus.

	out_of_domain_classifier: nesta opção será colocado o identificador do classificador a usar para decidir se as interações recebidas são ou não do domínio de aplicação. Se a opção out_of_domain_corpus estiver vazia, este parâmetro não é considerado.
Para os domínios do corpo AIA-BDE, estão disponíveis três classificadores pré-treinados, com os seguintes identificadores: "SVM", "NB", "RF".

	decision: no caso de se lançar mais de um agente em paralelo (ver opção agents), a escolha da melhor resposta pode seguir uma estratégia de decisão, a identificar nesta opção. Estão disponíveis duas estratégias de definição: caso esta opção nomeadamente Majority Voting, se esta opção estiver vazia, ou Borda Count, se o valor for "borda".

	number_of_answers_per_agent: ao se lançar mais do que um agente em paralelo (opção agents) e escolher uma estratégia de decisão (opção decision), podem considerar-se as n primeiras respostas dadas por cada agente. O valor de n deve ser colocado nesta opção. É de notar que este valor não é o número de respostas apresentado ao utilizador, mas o número de respostas que irão ser ponderadas na decisão sobre as respostas a apresentar.

	theta: todos os agentes já implementados se baseiam num ranking de respostas. Para maximizar a possibilidade de encontrar a resposta desejada, será possível tirar partido desse ranking e, em alguns casos, apresentar mais do que uma resposta. O valor de theta indica a distância máxima a que as respostas adicionais podem estar da primeira resposta do ranking. Para retornar apenas uma resposta, este opção deve ter o valor 0.

	w2v_embeddings: esta opção é usada apenas pelo agente baseado num modelo word2vec e indica a localização desse modelo na máquina local.
	
	bert_embeddings: esta opção é usada apenas pelo agente baseado num modelo BERT e indica o identificador desse modelo na biblioteca python Transformers.
	

## Utilização. 

O sistema AMAIA pode ser configurado para ser utilizado de três modos.

Para usar através de numa linha de comandos:

> python3 controler.py

Para usar integrado numa página web, através de uma interface baseada em Flask:

> python3 run_in_flask.py

Por defeito, esta aplicação irá estar disponível localmente, através do porto 5001 (http://localhost:5001/), opções que podem ser alteradas no código do ficheiro run_in_flask.py

Para usar através do Slack:

> python3 run_in_slack.py

Para este último será necessário criar uma app para o workspace onde o sistema será utilizado, de forma a obter o Signing Secret (SLACK_SIGNING_SECRET) e o Bot User OAuth Access Token (SLACK_BOT_TOKEN) necessários. Para a informação mais atualizada sobre como criar uma app deverá ser consultado o site https://api.slack.com.


## Arquitetura

Um visão alto nível da arquitetura do sistema está representada na imagem arquitetura-AMAIA.png.

A criação de um novo agente implica:
    
    a) criar uma classe que descenda da classe DefaultAgent e que re-implemente o método matching_questions();
    
    b) adicionar à função createSingleAgent() do ficheiro GeneralAgent.py uma entrada que associe um ID ao novo agente.

Se for necessário, podem adicionar-se opções ao ficheiro config.txt, a tratar no ficheiro controller.py

Os classificadores também podem voltar a ser treinados, usando os scripts Python na diretoria Classificador.
