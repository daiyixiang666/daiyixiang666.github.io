export const site = {
  title: 'Yixiang / Notes',
  author: 'Yixiang',
  description: '关于机器学习、强化学习与生成模型的学习笔记。把公式、代码和直觉连起来。',
  github: 'https://github.com/daiyixiang666',
};
export function readingTime(body = '') {
  const chinese = body.match(/[\u3400-\u9fff]/g)?.length ?? 0;
  const words = body.replace(/[\u3400-\u9fff]/g, '').split(/\s+/).filter(Boolean).length;
  return Math.max(1, Math.ceil(chinese / 300 + words / 200));
}
export function formatDate(date: Date) {
  return date.toISOString().slice(0, 10).replaceAll('-', '.');
}
