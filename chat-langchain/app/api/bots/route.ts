import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";

const getBots = () => {
  const botsDir = path.resolve(process.cwd(), "public", "bots");
  const botFiles = fs.readdirSync(botsDir);
  const botNames = botFiles
    .map((file) => path.parse(file).name)
    .filter((value, index, self) => self.indexOf(value) === index);

  const bots = botNames.filter((botName) => {
    const hasMp3 = botFiles.includes(`${botName}.mp3`);
    const hasJpg = botFiles.includes(`${botName}.jpg`);
    return hasMp3 && hasJpg;
  });

  return bots;
};
export async function GET() {
  const bots = getBots();
  return NextResponse.json({ bots });
}
