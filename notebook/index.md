---
layout: page
title: Notebook
description: Pavans notebook
permalink: /notebook/
---

Initially I blogged to add a drop in ocean of knowledge that internet has and felt content when post helped solve others problem. But honestly, writing a single blog post takes more than 4 hours and then youtube happened, I spoke my thoughts out in no time. But this notebook is different, I am writing notes to keep track of my learning and also help me not to forget what I learned.  

<ul>
  {% for post in site.categories.notebook %}
    <li>
        <span>{{ post.date | date_to_string }}</span> » <a href="{{ post.url }}" title="{{ post.title }}">{{ post.title }}</a>
    </li>
  {% endfor %}
</ul>