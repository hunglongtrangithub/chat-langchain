FROM node:18-alpine

ARG NEXT_PUBLIC_API_BASE_URL
ENV NEXT_PUBLIC_API_BASE_URL=$NEXT_PUBLIC_API_BASE_URL

WORKDIR /app

COPY package*.json yarn.lock ./
RUN yarn install --production

COPY .en[v] ./
COPY . .

RUN yarn build

EXPOSE 3000
CMD ["yarn", "start"]
