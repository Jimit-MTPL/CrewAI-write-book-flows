from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool
#from langchain_openai import ChatOpenAI

from write_a_book_with_flows.types import BookOutline
from langchain_groq import ChatGroq

@CrewBase
class OutlineCrew:
    """Book Outline Crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"
    #llm = ChatGroq(model="gemini/gemini-pro")

    @agent
    def researcher(self) -> Agent:
        search_tool = SerperDevTool(search_url="https://google.serper.dev/scholar")
        return Agent(
            config=self.agents_config["researcher"],
            tools=[search_tool],
            llm="gemini/gemini-1.5-flash",
            verbose=True,
        )

    @agent
    def outliner(self) -> Agent:
        return Agent(
            config=self.agents_config["outliner"],
            llm="gemini/gemini-1.5-flash",
            verbose=True,
        )

    @task
    def research_topic(self) -> Task:
        return Task(
            config=self.tasks_config["research_topic"],
        )

    @task
    def generate_outline(self) -> Task:
        return Task(
            config=self.tasks_config["generate_outline"], output_pydantic=BookOutline
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Book Outline Crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
