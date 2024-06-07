"use client";

import React, { useRef, useEffect, useState } from "react";
import { v4 as uuidv4 } from "uuid";
import { EmptyState } from "../components/EmptyState";
import { ChatMessageBubble, Message } from "../components/ChatMessageBubble";
import { AutoResizeTextarea } from "./AutoResizeTextarea";
import { marked } from "marked";
import { Renderer } from "marked";
import hljs from "highlight.js";
import "highlight.js/styles/gradient-dark.css";

import { fetchEventSource } from "@microsoft/fetch-event-source";
import { applyPatch } from "fast-json-patch";
import { toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import {
  Heading,
  Flex,
  IconButton,
  InputGroup,
  InputRightElement,
  InputLeftElement,
  Spinner,
  CircularProgress,
  Avatar,
  HStack,
  VStack,
  Button,
  Menu,
  MenuButton,
  MenuList,
  MenuItem,
} from "@chakra-ui/react";
import { ArrowUpIcon, ChevronDownIcon } from "@chakra-ui/icons";
import { MdMic, MdStop } from "react-icons/md";

import { Source } from "./SourceBubble";
import { apiBaseUrl } from "../utils/constants";
import { RiRobot2Line } from "react-icons/ri";
import Image from "next/image";

export function ChatWindow(props: { titleText?: string }) {
  const { titleText = "Medical Chatbot" } = props;
  const conversationId = uuidv4();
  const messageContainerRef = useRef<HTMLDivElement | null>(null);
  const [messages, setMessages] = useState<Array<Message>>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isSpeechLoading, setIsSpeechLoading] = useState(false);
  const [isSpeechPlaying, setIsSpeechPlaying] = useState(false);
  const [isVideoPlaying, setIsVideoPlaying] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [chatHistory, setChatHistory] = useState<
    { human: string; ai: string }[]
  >([]);
  const [chatbots, setChatbots] = useState<string[]>([]);
  const [selectedChatbot, setSelectedChatbot] = useState<string>("");

  const fetchBots = async () => {
    const response = await fetch("api/bots").then((res) => res.json());
    const botNames = response.bots;
    console.log("Available chatbots:", botNames);
    setChatbots(botNames);
  };
  // Load available chatbots on component mount
  useEffect(() => {
    fetchBots();
  }, []);

  // Create a ref for selected chatbot state
  const selectedChatbotRef = useRef(selectedChatbot);
  useEffect(() => {
    selectedChatbotRef.current = selectedChatbot;
  }, [selectedChatbot]);

  const videoRef = useRef<HTMLVideoElement | null>(null);

  const recorderRef = useRef({
    audioChunks: [] as Blob[],
    mediaRecorder: null as MediaRecorder | null,

    start: async function () {
      if (!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia)) {
        return Promise.reject(
          new Error(
            "mediaDevices API or getUserMedia method is not supported in this browser.",
          ),
        );
      }
      console.log("start recording");
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      this.mediaRecorder = new MediaRecorder(stream);

      this.mediaRecorder.ondataavailable = (event) => {
        this.audioChunks.push(event.data);
        console.log("Chunk length: ", this.audioChunks.length);
      };
      this.mediaRecorder.start(10); // HACK: This is a temporary fix to prevent the first chunk from being empty
    },

    stop: async function (callback_action: null | "audio" | "video" = null) {
      console.log("stop recording");
      if (this.mediaRecorder) {
        this.mediaRecorder.onstop = async () => {
          const audioBlob = new Blob(this.audioChunks, { type: "audio/mp3" });
          this.mediaRecorder?.stream
            .getTracks()
            .forEach((track) => track.stop()); // Stop all tracks
          this.audioChunks = []; // Clear audioChunks after stopping
          this.mediaRecorder = null; // Reset mediaRecorder after stopping

          await transcribeAndSendMessage(audioBlob, callback_action);
        };
        this.mediaRecorder.stop();
      }
    },
  });

  const sendMessage = async (
    message?: string,
    noStream: boolean = false,
  ): Promise<string | undefined> => {
    console.log("API Base Url:", apiBaseUrl);
    if (messageContainerRef.current) {
      messageContainerRef.current.classList.add("grow");
    }
    if (isLoading) {
      return;
    }
    const messageValue = message ?? input;
    if (messageValue === "") return;
    setInput("");
    setMessages((prevMessages) => [
      ...prevMessages,
      {
        id: Math.random().toString(),
        content: messageValue,
        text: messageValue,
        role: "user",
      },
    ]);
    setIsLoading(true);

    let accumulatedMessage = "";
    let runId: string | undefined = undefined;
    let sources: Source[] | undefined = undefined;
    let messageIndex: number | null = null;

    let renderer = new Renderer();
    renderer.paragraph = (text) => {
      return text + "\n";
    };
    renderer.list = (text) => {
      return `${text}\n\n`;
    };
    renderer.listitem = (text) => {
      return `\n• ${text}`;
    };
    renderer.code = (code, language) => {
      const validLanguage = hljs.getLanguage(language || "")
        ? language
        : "plaintext";
      const highlightedCode = hljs.highlight(
        validLanguage || "plaintext",
        code,
      ).value;
      return `<pre class="highlight bg-gray-700" style="padding: 5px; border-radius: 5px; overflow: auto; overflow-wrap: anywhere; white-space: pre-wrap; max-width: 100%; display: block; line-height: 1.2"><code class="${language}" style="color: #d6e2ef; font-size: 12px; ">${highlightedCode}</code></pre>`;
    };
    marked.setOptions({ renderer });
    try {
      const sourceStepName = "FindDocs";
      let streamedResponse: Record<string, any> = {};
      if (noStream) {
        await fetch(apiBaseUrl + "/chat/ainvoke_log", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            question: messageValue,
            chat_history: chatHistory,
          }),
        })
          .then((response) => {
            console.log(response);
            if (!response.ok) {
              throw new Error("Failed to fetch response from the server.");
            }
            return response.json();
          })
          .then((data) => {
            console.log(data);
            accumulatedMessage = data.response;
            const parsedResult = marked.parse(data.response);
            setMessages((prevMessages) => {
              let newMessages = [...prevMessages];
              newMessages.push({
                id: Math.random().toString(),
                content: parsedResult.trim(),
                text: data.response.trim(),
                runId: runId,
                sources: data.docs.map((doc: string) => ({
                  url: JSON.parse(doc).source,
                  title: JSON.parse(doc).title,
                })),
                role: "assistant",
              });
              return newMessages;
            });
          });
        setChatHistory((prevChatHistory) => [
          ...prevChatHistory,
          { human: messageValue, ai: accumulatedMessage },
        ]);
        setIsLoading(false);
      } else {
        await fetchEventSource(apiBaseUrl + "/chat/astream_log", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Accept: "text/event-stream",
          },
          body: JSON.stringify({
            question: messageValue,
            chat_history: chatHistory,
          }),
          openWhenHidden: true,
          onerror(err) {
            throw err;
          },
          onmessage(msg) {
            if (msg.event === "end") {
              setChatHistory((prevChatHistory) => [
                ...prevChatHistory,
                { human: messageValue, ai: accumulatedMessage },
              ]);
              setIsLoading(false);
            }
            if (msg.event === "error") {
              let errorMessage = JSON.parse(msg.data).message;
              toast.error(errorMessage);
            }
            if (msg.event === "data" && msg.data) {
              const chunk = JSON.parse(msg.data);
              streamedResponse = applyPatch(
                streamedResponse,
                chunk.ops,
              ).newDocument;
              if (
                Array.isArray(
                  streamedResponse?.logs?.[sourceStepName]?.final_output
                    ?.output,
                )
              ) {
                sources = streamedResponse.logs[
                  sourceStepName
                ].final_output.output.map((doc: string) => ({
                  url: JSON.parse(doc).source,
                  title: JSON.parse(doc).title,
                }));
              }
              if (streamedResponse.id !== undefined) {
                runId = streamedResponse.id;
              }
              if (Array.isArray(streamedResponse?.streamed_output)) {
                accumulatedMessage = streamedResponse.streamed_output.join("");
              }
              const parsedResult = marked.parse(accumulatedMessage);

              setMessages((prevMessages) => {
                let newMessages = [...prevMessages];
                if (
                  messageIndex === null ||
                  newMessages[messageIndex] === undefined
                ) {
                  messageIndex = newMessages.length;
                  newMessages.push({
                    id: Math.random().toString(),
                    content: parsedResult.trim(),
                    text: accumulatedMessage.trim(),
                    runId: runId,
                    sources: sources,
                    role: "assistant",
                  });
                } else if (newMessages[messageIndex] !== undefined) {
                  newMessages[messageIndex].content = parsedResult.trim();
                  newMessages[messageIndex].text = accumulatedMessage.trim();
                  newMessages[messageIndex].runId = runId;
                  newMessages[messageIndex].sources = sources;
                }
                return newMessages;
              });
            }
          },
        });
      }
      return accumulatedMessage;
    } catch (e: any) {
      toast.error(e.toString());
      setMessages((prevMessages) => prevMessages.slice(0, -1));
      setIsLoading(false);
      setInput(messageValue);
      throw e;
    }
  };

  const sendInitialQuestion = async (question: string) => {
    await sendMessage(question);
  };

  const playMessageAudio = async (message: string, selectedChatbot: string) => {
    console.log("play message audio");
    const formData = new FormData();
    const botAudioBlob = await fetch(`/bots/${selectedChatbot}.mp3`).then(
      (res) => res.blob(),
    );
    formData.append("bot_voice_file", botAudioBlob, `${selectedChatbot}.mp3`);
    formData.append("message", message);
    formData.append("conversationId", conversationId);
    formData.append("chatbot", selectedChatbot);
    setIsSpeechLoading(true);
    const audioResponse = await fetch(apiBaseUrl + "/text_to_speech", {
      method: "POST",
      body: formData,
    });
    if (!audioResponse.ok) {
      const errorMessage = await audioResponse.text();
      console.error("Failed to fetch audio:", errorMessage);
      toast.error("Failed to transform text to speech for AI response:" + errorMessage);

      setIsSpeechLoading(false);
      return;
    }

    const audioBlob = await audioResponse.blob();
    const audioUrl = URL.createObjectURL(audioBlob);
    const audio = new Audio(audioUrl);

    console.log("playing audio");
    audio.play();

    audio.onplay = () => {
      console.log("audio playing");
      setIsSpeechLoading(false);
      setIsSpeechPlaying(true);
    };

    audio.onended = () => {
      console.log("audio ended");
      setIsSpeechPlaying(false);
    };
  };

  const playMessageVideo = async (message: string, selectedChatbot: string) => {
    console.log("play speech video");

    let formData = new FormData();
    const botAudioBlob = await fetch(`/bots/${selectedChatbot}.mp3`).then(
      (res) => res.blob(),
    );
    formData.append("bot_voice_file", botAudioBlob, `${selectedChatbot}.mp3`);
    formData.append("message", message);
    formData.append("conversationId", conversationId);
    formData.append("chatbot", selectedChatbot);
    setIsSpeechLoading(true);
    const audioResponse = await fetch(apiBaseUrl + "/text_to_speech", {
      method: "POST",
      body: formData,
    });
    if (!audioResponse.ok) {
      const errorMessage = await audioResponse.text();
      console.error("Failed to fetch audio:", errorMessage);
      toast.error("Failed to transform text to speech for AI response:" + errorMessage);

      setIsSpeechLoading(false);
      return;
    }
    const audioBlob = await audioResponse.blob();

    formData = new FormData();
    formData.append("bot_speech_file", audioBlob, `${selectedChatbot}.mp3`);
    const botImageBlob = await fetch(`/bots/${selectedChatbot}.jpg`).then(
      (res) => res.blob(),
    );
    formData.append("bot_image_file", botImageBlob, `${selectedChatbot}.jpg`);
    formData.append("conversationId", conversationId);
    formData.append("chatbot", selectedChatbot);
    const videoResponse = await fetch(apiBaseUrl + "/speech_to_video", {
      method: "POST",
      body: formData,
    });
    if (!videoResponse.ok) {
      const errorMessage = await videoResponse.text();
      console.error("Failed to fetch video:", errorMessage);
      toast.error("Failed to transform text to video for AI response:" + errorMessage);

      setIsSpeechLoading(false);
      return;
    }
    setIsVideoPlaying(true);
    const videoBlob = await videoResponse.blob();

    const videoUrl = URL.createObjectURL(videoBlob);
    if (videoRef.current) {
      videoRef.current.src = videoUrl;
      videoRef.current.play();

      videoRef.current.onplay = () => {
        console.log("video playing");
        setIsSpeechLoading(false);
        setIsSpeechPlaying(true);
      };

      videoRef.current.onended = () => {
        console.log("video ended");
        setIsSpeechPlaying(false);
        setIsVideoPlaying(false);
      };
    }
  };

  const transcribeAndSendMessage = async (
    audioBlob: Blob,
    callback_action: null | "audio" | "video" = null,
  ) => {
    console.log("transcribe and send message");

    const formData = new FormData();
    formData.append("user_audio_file", audioBlob, `${conversationId}.mp3`);
    formData.append("conversationId", conversationId);

    setIsTranscribing(true);
    const response = await fetch(apiBaseUrl + "/transcribe_audio", {
      method: "POST",
      body: formData,
    });

    if (response.ok) {
      const data = await response.json();
      console.log(data);
      const userMessage = data.transcript;
      setIsTranscribing(false);
      sendMessage(userMessage).then((aiMessage) => {
        if (!aiMessage) return;
        console.log("AI Message:", aiMessage);
        if (callback_action === "audio") {
          playMessageAudio(aiMessage, selectedChatbotRef.current);
        } else if (callback_action === "video") {
          playMessageVideo(aiMessage, selectedChatbotRef.current);
        }
      });
    } else {
      setIsTranscribing(false);
      toast.error("Failed to transform user audio to text.");
    }
  };

  return (
    <div className="flex flex-col items-center p-8 rounded grow max-h-full">
      {messages.length > 0 && (
        <Flex direction={"column"} alignItems={"center"} paddingBottom={"20px"}>
          <Heading fontSize="2xl" fontWeight={"medium"} mb={1} color={"white"}>
            {titleText}
          </Heading>
          <Heading fontSize="md" fontWeight={"normal"} mb={1} color={"white"}>
            We appreciate feedback!
          </Heading>
        </Flex>
      )}
      <HStack spacing={400}>
        <VStack spacing={10}>
          {isVideoPlaying ? (
            <video
              ref={videoRef}
              style={{
                width: "150px",
                height: "150px",
                objectFit: "cover",
                borderRadius: "50%",
                border: "3px solid gray",
              }}
              autoPlay
            >
              Your browser does not support the video tag.
            </video>
          ) : (
            <Avatar
              style={{
                width: "150px",
                height: "150px",
                objectFit: "cover",
                borderRadius: "50%",
                border: "3px solid gray",
              }}
              name="ChatBot"
              src={
                selectedChatbot !== ""
                  ? `/bots/${selectedChatbot}.jpg`
                  : "/images/bot.png"
              }
            />
          )}
          <IconButton
            colorScheme="blue"
            rounded={"full"}
            aria-label="Bot status"
            icon={
              isSpeechPlaying ? (
                <Spinner />
              ) : isSpeechLoading ? (
                <CircularProgress
                  isIndeterminate
                  size="30px"
                  color="blue.300"
                />
              ) : (
                <RiRobot2Line />
              )
            }
          />
        </VStack>
        <VStack spacing={10}>
          <Avatar
            name="User"
            style={{
              width: "150px",
              height: "150px",
              objectFit: "cover",
              borderRadius: "50%",
              border: "3px solid gray",
            }}
            src="/images/user.png"
          />
          <IconButton
            colorScheme="blue"
            rounded={"full"}
            aria-label="User mic status"
            icon={
              isRecording ? (
                <MdStop />
              ) : isTranscribing ? (
                <Spinner />
              ) : (
                <MdMic />
              )
            }
            type="submit"
            onClick={(e) => {
              e.preventDefault();
              if (isRecording) {
                setIsRecording(false);
                recorderRef.current.stop("video");
              } else {
                setIsRecording(true);
                recorderRef.current.start();
              }
            }}
          />
        </VStack>
      </HStack>
      <Menu>
        <MenuButton as={Button} rightIcon={<ChevronDownIcon />}>
          {selectedChatbot || "Select Chatbot"}
        </MenuButton>
        <MenuList>
          {chatbots.map((bot) => (
            <MenuItem
              key={bot}
              onClick={() => {
                console.log("Selected chatbot: ", bot);
                setSelectedChatbot(bot);
              }}
            >
              {bot}
            </MenuItem>
          ))}
        </MenuList>
      </Menu>
      <div
        className="flex flex-col-reverse w-full mb-2 overflow-auto"
        ref={messageContainerRef}
      >
        {messages.length > 0 ? (
          [...messages]
            .reverse()
            .map((m, index) => (
              <ChatMessageBubble
                key={m.id}
                message={{ ...m }}
                aiEmoji="🦜"
                isMostRecent={index === 0}
                messageCompleted={!isLoading}
              ></ChatMessageBubble>
            ))
        ) : (
          <EmptyState onChoice={sendInitialQuestion} titleText={titleText} />
        )}
      </div>
      <InputGroup size="md" alignItems={"center"}>
        <InputLeftElement h="full">
          <IconButton
            colorScheme="blue"
            rounded={"full"}
            aria-label="Send"
            icon={
              isRecording ? (
                <MdStop />
              ) : isTranscribing ? (
                <Spinner />
              ) : (
                <MdMic />
              )
            }
            type="submit"
            onClick={(e) => {
              e.preventDefault();
              if (isRecording) {
                setIsRecording(false);
                recorderRef.current.stop();
              } else {
                setIsRecording(true);
                recorderRef.current.start();
              }
            }}
          />
        </InputLeftElement>
        <AutoResizeTextarea
          value={input}
          maxRows={5}
          marginRight={"56px"}
          marginLeft={"56px"}
          placeholder="What is LangChain Expression Language?"
          textColor={"white"}
          borderColor={"rgb(58, 58, 61)"}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              sendMessage();
            } else if (e.key === "Enter" && e.shiftKey) {
              e.preventDefault();
              setInput(input + "\n");
            }
          }}
        />
        <InputRightElement h="full">
          <IconButton
            colorScheme="blue"
            rounded={"full"}
            aria-label="Send"
            icon={isLoading ? <Spinner /> : <ArrowUpIcon />}
            type="submit"
            onClick={(e) => {
              e.preventDefault();
              sendMessage();
            }}
          />
        </InputRightElement>
      </InputGroup>

      {messages.length === 0 ? (
        <footer className="flex justify-center absolute bottom-8">
          <a
            href="hhttps://github.com/hunglongtrangithub/chat-langchain"
            target="_blank"
            className="text-white flex items-center"
          >
            <Image
              src="/images/github-mark.svg"
              alt="Github Logo"
              width={20}
              height={20}
            />
            <span>View Source</span>
          </a>
        </footer>
      ) : (
        ""
      )}
    </div>
  );
}
