# Yixiang / Notes

个人技术博客，使用 Astro、Markdown 和 GitHub Pages。站点地址：https://daiyixiang666.github.io

## 本地使用

需要 Node.js 24。

```sh
npm ci
npm run dev
```

打开终端打印的本地地址。`npm run build` 生成静态网站，`npm run preview` 预览构建结果。

## 写文章

在 `src/content/posts/` 新建一个 `.md` 文件，例如 `my-first-post.md`：

```markdown
---
title: "我的第一篇文章"
description: "一句话介绍这篇文章。"
date: 2026-11-01
category: "学习笔记"
cover: "/covers/latent.svg"
coverAlt: "潜在空间示意图"
draft: true
---

## 从一个问题开始

正文写在这里。
```

- `draft: true` 不会生成文章页面，不出现在首页、RSS 和站点地图中。**公开仓库内的草稿源码仍然公开。** 私密草稿应保存在仓库外。
- 准备发布时改为 `draft: false`，提交并推送到 `main`，GitHub Actions 会自动部署。
- 支持 `$...$` 行内公式、`$$...$$` 独立公式，以及带语言标记的代码块。
- 封面放在 `public/covers/`，引用路径为 `/covers/文件名`。
- 当前三篇 `sample: true` 的文章是演示内容，首页和正文均有标记。可删除或替换。
- 站名、描述和 GitHub 链接在 `src/site.ts`；关于页面在 `src/pages/about.astro`。
- 更换域名时，同时修改 `astro.config.mjs` 的 `site` 和 `public/robots.txt` 的站点地图地址。

## 部署

仓库 Settings → Pages → Source 设为 GitHub Actions。工作流位于 `.github/workflows/deploy.yml`。

仅这个独立项目用于公开发布，不包含原有笔记、简历或其他仓库文件。
