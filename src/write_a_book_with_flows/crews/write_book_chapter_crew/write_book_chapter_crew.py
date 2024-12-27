from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool
#from langchain_openai import ChatOpenAI

from write_a_book_with_flows.types import Chapter
from langchain_groq import ChatGroq

@CrewBase
class WriteBookChapterCrew:
    """Write Book Chapter Crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"
    #llm = ChatGroq(model="groq/llama-3.3-70b-versatile")

    @agent
    def researcher(self) -> Agent:
        search_tool = SerperDevTool(search_url="https://google.serper.dev/scholar")
        return Agent(
            config=self.agents_config["researcher"],
            tools=[search_tool],
            llm="gemini/gemini-1.5-flash",
            verbose=True
        )

    @agent
    def writer(self) -> Agent:
        return Agent(
            config=self.agents_config["writer"],
            llm="gemini/gemini-1.5-flash",
        )

    @task
    def research_chapter(self) -> Task:
        return Task(
            config=self.tasks_config["research_chapter"],
        )

    @task
    def write_chapter(self) -> Task:
        return Task(config=self.tasks_config["write_chapter"], output_pydantic=Chapter)

    @crew
    def crew(self) -> Crew:
        """Creates the Write Book Chapter Crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )