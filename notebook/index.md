---
layout: page
title: Notebook
description: Pavans notebook
permalink: /notebook/
---

Initially I blogged to add a drop in ocean of knowledge that internet has and felt content when post helped solve others problem. But honestly, writing a single blog post takes more than 4 hours and then youtube happened, I spoke my thoughts out and saved time by not writing. But now situation is different, I am writing notes to remember important points from my learnings.  

<ul>
  {% for post in site.categories.notebook %}
    <li>
        <span>{{ post.date | date_to_string }}</span> » <a href="{{ post.url }}" title="{{ post.title }}">{{ post.title }}</a>
    </li>
  {% endfor %}
</ul>