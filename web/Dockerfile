# syntax=docker/dockerfile:1
# Build context is the `web/` directory.

FROM node:22-alpine AS build
WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY tsconfig.json vite.config.ts tailwind.config.js postcss.config.js index.html ./
COPY src ./src
# Static files served as-is: the mail logo and the favicon. Omitting this is why
# https://.../logo-mail.png answered with the SPA's index.html, and the welcome
# email's header image never rendered.
COPY public ./public
RUN npm run build

FROM nginx:alpine AS runtime
RUN apk add --no-cache wget
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD wget -qO- http://127.0.0.1/healthz || exit 1
