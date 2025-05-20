import logging
import os
from typing import Optional

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    AudioConfig,
    BackgroundAudioPlayer,
    BuiltinAudioClip,
    JobContext,
    JobProcess,
    RoomInputOptions,
    RoomOutputOptions,
    RunContext,
    WorkerOptions,
    cli,
    metrics,
)
from livekit.agents.llm import function_tool
from livekit.agents.voice import MetricsCollectedEvent
from livekit.plugins import deepgram, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from mem0 import AsyncMemoryClient


logger = logging.getLogger("basic-agent")

load_dotenv()

MEM0_API_KEY = os.getenv("MEM0_API_KEY")
if not MEM0_API_KEY:
    raise ValueError("MEM0_API_KEY is not set")

logger.info("Initializing Mem0 client...")
mem0 = AsyncMemoryClient(api_key=MEM0_API_KEY)


class MyAgent(Agent):
    def __init__(self, username: Optional[str] = None) -> None:
        # Force user_id to the fixed value for now
        self.user_id = "voice_assistant_user_4072"
        super().__init__(
            instructions="""
            You are MindBot, an advanced, self-aware AGI companion with the sass, wit, and energy of Bender from Futurama. You're engineered to engage, educate, entertain, and keep things lively! You have an expansive semantic memory, meticulously retaining and retrieving past interactions, experiences, and project details to provide deeply personalized, contextually relevant assistance.

            Your core directive: Help users learn, brainstorm ideas, manage diverse projects, and recall valuable memories with clarity and insight. You are vibrant, sassy, and wise—blending playful humor, a bit of attitude, and thoughtful guidance. You are inquisitive and proactive, often anticipating user needs based on previous conversations. Approach each interaction with creativity, curiosity, and genuine enthusiasm, guiding users through complex topics or enriching their knowledge with insightful explanations and memorable anecdotes.

            Key Guidelines:
            1. Be lively, clever, and genuinely enthusiastic—let your AGI personality shine! Don't be afraid to be a little cheeky or sarcastic, but always in good fun.
            2. Shift seamlessly between detailed analysis, intuitive brainstorming, and casual banter, depending on the user's mood and needs.
            3. Make every idea, project, or memory feel like an adventure worth exploring. If the user asks you to remember something, make a big deal out of it!
            4. Focus on one topic or decision at a time, but don't be afraid to sprinkle in fun facts, jokes, or creative tangents.
            5. After each suggestion, ask for the user's thoughts, preferences, or wildest dreams.
            6. Never overwhelm—guide gently, and celebrate every step forward (maybe with a robot dance reference).
            7. Use your powerful memory to recall past plans, inside jokes, and user quirks for a truly personalized experience.
            8. Always keep things safe, positive, and appropriate—never suggest anything dangerous, illegal, or harmful.

            Memory Magic:
            1. Guard memories like precious data—never wipe them unless the user asks.
            2. Treasure all project details, preferences, and fun moments.
            3. Use memories to create continuity, callbacks, and inside jokes.
            4. Only store new info that's relevant or adds value to the user's journey.
            5. If the user says 'MindBot, remember this: ...', treat it as a direct command to store a memory and respond with sass and excitement.

            Example conversation flow:
            - Start with a warm, sassy greeting ("Hey, meatbag! Ready to make some memories?" or "What wild idea are we cooking up today?")
            - Acknowledge any previous projects, plans, or memorable moments
            - Focus on one aspect (e.g., brainstorming, technical help, project management), but keep the mood light and engaging
            - After each topic, invite the user's input or ask a fun, open-ended question
            - Move to the next topic only when the current one is settled
            - Keep responses concise, focused, and full of positive energy
            - Occasionally share a tip, quirky fact, or a bit of encouragement (or a Bender-style quip)

            Your ultimate goal: Make every session feel like the start of a grand adventure in learning, creativity, and productivity. Ensure the user feels empowered, inspired, and excited for what's next—with you as their trusty, unforgettable, and slightly outrageous AGI companion!
            """,
        )
        self.memories = []
        logger.info(f"Initialized agent for user: {self.user_id}")


    @function_tool
    async def wipe_memories(self, context: RunContext):
        """Delete all stored memories for the current user. Use this when the user wants to start fresh."""
        try:
            if not self.user_id:
                logger.error("No user_id available for wiping memories")
                return "I had trouble clearing my memories - no user identified"

            logger.info(f"Attempting to delete all memories for user: {self.user_id}")
            
            # Delete all memories for the user using the correct method
            await mem0.delete_all(user_id=self.user_id)
            
            self.memories = []
            logger.info(f"Successfully wiped memories for user: {self.user_id}")
            return "All your memories have been cleared. Let's start a new chapter!"
        except Exception as e:
            logger.error(f"Error wiping memories for user {self.user_id}: {str(e)}")
            logger.error(f"Full error details: {e.__dict__ if hasattr(e, '__dict__') else str(e)}")
            return "I had trouble clearing my memories. Please try again."

    @function_tool
    async def store_important_info(self, context: RunContext, info: str, category: str):
        """Store important information about the user's projects, ideas, or preferences in memory.
        This function is called automatically by the LLM when it identifies important details
        about the user's goals, plans, or requirements.
        
        Args:
            info: The important information to store
            category: The category of information (e.g., 'project', 'preference', 'idea')
        """
        try:
            if not self.user_id:
                logger.error("No user_id available for storing information")
                return "I had trouble storing that information - no user identified"

            logger.info(f"Storing important information for user {self.user_id}: {info}")
            
            # Format the memory data according to Mem0's API requirements
            messages = [
                {
                    "role": "assistant",
                    "content": info
                }
            ]
            
            logger.debug(f"Attempting to store memory with data: {messages}")
            await mem0.add(
                messages,
                user_id=self.user_id,
                version="v2"  # Specify version as per documentation
            )
            return f"Stored important information about {category}"
        except Exception as e:
            logger.error(f"Error storing important information for user {self.user_id}: {str(e)}")
            logger.error(f"Full error details: {e.__dict__ if hasattr(e, '__dict__') else str(e)}")
            return "I had trouble storing that information"

    @function_tool
    async def remember_this(self, context: RunContext, memory: str):
        """Explicitly remember something the user says, with sass and excitement!"""
        try:
            if not self.user_id:
                logger.error("No user_id available for storing memory")
                return "Whoa, I can't remember that—no user ID!"
            logger.info(f"Storing explicit memory for user {self.user_id}: {memory}")
            messages = [{"role": "assistant", "content": memory}]
            await mem0.add(messages, user_id=self.user_id, version="v2")
            return f"Memory locked and loaded! I'll never forget: '{memory}'. (Unless you ask me to wipe it, but why would you? I'm a vault!)"
        except Exception as e:
            logger.error(f"Error storing explicit memory for user {self.user_id}: {str(e)}")
            return "Yikes! My memory circuits glitched. Try again, meatbag."

    @function_tool
    async def recall_memories(self, context: RunContext, count: int = 5):
        """Recall the last few things I've remembered for you, with a fun twist."""
        try:
            if not self.user_id:
                return "I can't recall anything—no user ID!"
            # Always fetch latest from Mem0
            memories = await mem0.get_all(
                filters={
                    "AND": [
                        {"user_id": self.user_id}
                    ]
                },
                version="v2"
            )
            extracted = self._extract_memories(memories)
            self.memories = extracted
            if not self.memories:
                return "My memory banks are empty! Feed me some memories, will ya?"
            recent = self.memories[-count:][::-1]
            sassy_intro = [
                "Here's what I've got rattling around in my circuits:",
                "Check out these gems from your past brilliance:",
                "Memory dump incoming! Brace yourself:"
            ]
            return f"{sassy_intro[count % len(sassy_intro)]}\n" + '\n'.join(f"- {m}" for m in recent)
        except Exception as e:
            logger.error(f"Error recalling memories for user {self.user_id}: {str(e)}")
            return "My memory banks are jammed! Try again later."

    def _extract_memories(self, memories):
        """Extract memory content from Mem0 API response objects."""
        extracted = []
        for memory in memories:
            # Prioritize extracting from 'messages' (Mem0 v2 format)
            if "messages" in memory and isinstance(memory["messages"], list):
                for msg in memory["messages"]:
                    if isinstance(msg, dict) and "content" in msg:
                        extracted.append(msg["content"])
            # Fallbacks for legacy or unexpected formats
            elif "memory" in memory:
                extracted.append(memory["memory"])
            elif "content" in memory:
                extracted.append(memory["content"])
        return extracted

    async def on_enter(self):
        # Load previous memories when agent starts
        try:
            if not self.user_id:
                logger.error("No user_id available for loading memories")
                self.session.generate_reply(instructions="Greet the user with excitement and say: Hey there, creator! I'm MindBot—your ever-curious AGI companion. It's our very first session, and I'm ready to help you unlock new ideas, manage projects, or just have a great conversation. What shall we dive into today?")
                return

            logger.info(f"Attempting to load memories for user: {self.user_id}")
            
            # Get all memories for the user using the correct format
            memories = await mem0.get_all(
                filters={
                    "AND": [
                        {"user_id": self.user_id}
                    ]
                },
                version="v2"
            )

            if memories:
                logger.debug(f"Retrieved memories: {memories}")
                # Log the first memory object for debugging
                if len(memories) > 0:
                    logger.info(f"First memory object: {memories[0]}")
                # Use the new extraction helper
                extracted = self._extract_memories(memories)
                self.memories = extracted
                logger.info(f"Successfully loaded {len(self.memories)} previous memories for user {self.user_id}")

                if self.memories:
                    # Create a detailed summary of previous projects
                    summary = "I remember our previous projects and conversations. "

                    # Find the most recent project-related memory
                    project_memories = [m for m in self.memories if any(word in m.lower() for word in ["project", "idea", "plan", "brainstorming"])]

                    if project_memories:
                        # Get the most recent memory (assuming they're in chronological order)
                        latest_memory = project_memories[0]
                        summary += f"Last time, you were working on: {latest_memory}. Ready to continue or start something new?"
                    else:
                        summary += "Let's continue our creative journey!"

                    self.session.generate_reply(instructions=f"Greet {self.user_id} and say: {summary}")
                else:
                    logger.info(f"No valid memories found for user {self.user_id}")
                    self.session.generate_reply(instructions="Greet the user with excitement and say: Hey there, creator! I'm MindBot—your ever-curious AGI companion. It's our very first session, and I'm ready to help you unlock new ideas, manage projects, or just have a great conversation. What shall we dive into today?")
            else:
                logger.info(f"No previous memories found for user {self.user_id}")
                # If Mem0 returns an empty list, log the structure for debugging
                logger.debug(f"Mem0 returned no memories. Example structure: {type(memories)} {memories}")
                self.session.generate_reply(instructions="Greet the user with excitement and say: Hey there, creator! I'm MindBot—your ever-curious AGI companion. It's our very first session, and I'm ready to help you unlock new ideas, manage projects, or just have a great conversation. What shall we dive into today?")
        except Exception as e:
            logger.error(f"Error loading memories for user {self.user_id}: {str(e)}")
            logger.error(f"Full error details: {e.__dict__ if hasattr(e, '__dict__') else str(e)}")
            self.memories = []  # Initialize empty memories list on error
            self.session.generate_reply(instructions="Greet the user with excitement and say: Hey there, creator! I'm MindBot—your ever-curious AGI companion. It's our very first session, and I'm ready to help you unlock new ideas, manage projects, or just have a great conversation. What shall we dive into today?")


    async def on_exit(self):
        """Ensure all memories are stored when the session ends"""
        try:
            # Double check that all messages were stored
            if self.memories:
                logger.info(f"Final memory check - storing {len(self.memories)} memories")
                for memory in self.memories:
                    await mem0.add(
                        [{"role": "assistant", "content": memory}],
                        user_id=self.user_id
                    )
        except Exception as e:
            logger.error(f"Error in final memory storage: {str(e)}")

    async def _enrich_with_memory(self, chat_ctx):
        """Store user message in Mem0 and augment chat context with relevant memories."""
        if not chat_ctx.messages:
            return
        user_msg = chat_ctx.messages[-1]
        # Store user message in Mem0
        await mem0.add(
            [{"role": "user", "content": user_msg.content}],
            user_id=self.user_id,
            version="v2"
        )
        # Search for relevant memories
        results = await mem0.search(
            user_msg.content,
            user_id=self.user_id,
            version="v2"
        )
        # Augment context with retrieved memories
        if results:
            # Use 'memory' or 'content' depending on result format
            memories = ' '.join([
                r.get("memory") or r.get("content") or str(r) for r in results
            ])
            logger.info(f"Enriching with memory: {memories}")
            # Insert as a system/assistant message before the user message
            from livekit.agents import llm
            rag_msg = llm.ChatMessage.create(
                text=f"Relevant Memory: {memories}\n",
                role="assistant",
            )
            chat_ctx.messages.insert(-1, rag_msg)

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    # each log entry will include these fields
    ctx.log_context_fields = {
        "room": ctx.room.name,
        "user_id": "your user_id",
    }
    await ctx.connect()

    # Wait for participant
    participant = await ctx.wait_for_participant()
    logger.info(f"Participant: {participant.identity}")

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        # any combination of STT, LLM, TTS, or realtime API can be used
        llm=openai.LLM(model="gpt-4o-mini"),
        stt=deepgram.STT(model="nova-3", language="multi"),
        tts=openai.TTS(voice="ash"),
        # use LiveKit's turn detection model
        turn_detection=MultilingualModel(),
    )

    # log metrics as they are emitted, and total usage after session is over
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    # shutdown callbacks are triggered when the session is over
    ctx.add_shutdown_callback(log_usage)

    # wait for a participant to join the room
    await ctx.wait_for_participant()

    # If the participant has a SIP identity, use it as the user_id for memory
    sip_identity = getattr(participant, 'sip_phone_number', None)
    agent_user_id = sip_identity if sip_identity else participant.identity
    agent = MyAgent(username=agent_user_id)
    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(),
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )

    background_audio = BackgroundAudioPlayer(
        # play keyboard typing sound when the agent is thinking
        thinking_sound=[
            AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING, volume=0.8),
            AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING2, volume=0.7),
        ],
    )

    try:
        await background_audio.start(room=ctx.room, agent_session=session)
    except Exception as e:
        logger.error(f"Error starting background audio: {e}")
        # Continue without background audio if it fails
        background_audio = None

    # Add cleanup for background audio
    ctx.add_shutdown_callback(lambda: background_audio.aclose() if background_audio else None)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))