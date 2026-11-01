import rss from '@astrojs/rss';
import { getCollection } from 'astro:content';
import { site } from '../site';
import type { APIContext } from 'astro';
export async function GET(context: APIContext) {
  const posts = (await getCollection('posts', ({ data }) => !data.draft)).sort((a, b) => b.data.date.valueOf() - a.data.date.valueOf());
  return rss({ title: site.title, description: site.description, site: context.site!, items: posts.map(p => ({ title: p.data.title, description: p.data.description, pubDate: p.data.date, link: `/posts/${p.id}/` })), customData: '<language>zh-cn</language>' });
}
